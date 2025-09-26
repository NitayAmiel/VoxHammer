from pyexpat import model
from typing import *
import torch
import torch.nn as nn
import numpy as np
from transformers import CLIPTextModel, AutoTokenizer
import open3d as o3d
from .base import Pipeline
from . import samplers
from ..modules import sparse as sp
import pdb
from .. import attn_globals



class TrellisTextTo3DPipeline(Pipeline):
    """
    Pipeline for inferring Trellis text-to-3D models.

    Args:
        models (dict[str, nn.Module]): The models to use in the pipeline.
        sparse_structure_sampler (samplers.Sampler): The sampler for the sparse structure.
        slat_sampler (samplers.Sampler): The sampler for the structured latent.
        slat_normalization (dict): The normalization parameters for the structured latent.
        text_cond_model (str): The name of the text conditioning model.
    """
    def __init__(
        self,
        models: dict[str, nn.Module] = None,
        sparse_structure_sampler: samplers.Sampler = None,
        slat_sampler: samplers.Sampler = None,
        slat_normalization: dict = None,
        text_cond_model: str = None,
    ):
        if models is None:
            return
        super().__init__(models)
        self.sparse_structure_sampler = sparse_structure_sampler
        self.slat_sampler = slat_sampler
        self.sparse_structure_sampler_params = {}
        self.slat_sampler_params = {}
        self.slat_normalization = slat_normalization
        self._init_text_cond_model(text_cond_model)

    @staticmethod
    def from_pretrained(path: str) -> "TrellisTextTo3DPipeline":
        """
        Load a pretrained model.

        Args:
            path (str): The path to the model. Can be either local path or a Hugging Face repository.
        """
        pipeline = super(TrellisTextTo3DPipeline, TrellisTextTo3DPipeline).from_pretrained(path)
        new_pipeline = TrellisTextTo3DPipeline()
        new_pipeline.__dict__ = pipeline.__dict__
        args = pipeline._pretrained_args

        new_pipeline.sparse_structure_sampler = getattr(samplers, args['sparse_structure_sampler']['name'])(**args['sparse_structure_sampler']['args'])
        new_pipeline.sparse_structure_sampler_params = args['sparse_structure_sampler']['params']

        new_pipeline.slat_sampler = getattr(samplers, args['slat_sampler']['name'])(**args['slat_sampler']['args'])
        new_pipeline.slat_sampler_params = args['slat_sampler']['params']

        new_pipeline.slat_normalization = args['slat_normalization']

        new_pipeline._init_text_cond_model(args['text_cond_model'])

        return new_pipeline
    
    def _init_text_cond_model(self, name: str):
        """
        Initialize the text conditioning model.
        """
        # load model
        model = CLIPTextModel.from_pretrained(name)
        tokenizer = AutoTokenizer.from_pretrained(name)
        model.eval()
        model = model.cuda()
        self.text_cond_model = {
            'model': model,
            'tokenizer': tokenizer,
        }
        self.text_cond_model['null_cond'] = self.encode_text([''])

    @torch.no_grad()
    def encode_text(self, text: List[str]) -> torch.Tensor:
        """
        Encode the text.
        """
        assert isinstance(text, list) and all(isinstance(t, str) for t in text), "text must be a list of strings"
        encoding = self.text_cond_model['tokenizer'](text, max_length=77, padding='max_length', truncation=True, return_tensors='pt')
        tokens = encoding['input_ids'].cuda()
        embeddings = self.text_cond_model['model'](input_ids=tokens).last_hidden_state
        
        return embeddings
        
    def get_cond(self, prompt: List[str]) -> dict:
        """
        Get the conditioning information for the model.

        Args:
            prompt (List[str]): The text prompt.

        Returns:
            dict: The conditioning information
        """
        cond = self.encode_text(prompt)
        neg_cond = self.text_cond_model['null_cond']
        return {
            'cond': cond,
            'neg_cond': neg_cond,
        }
    def generate_and_save_attention(
        self,
        prompt_generation: str,
        prompt_attn: str,
        num_samples: int = 1,
        sampler_params: dict = {},
        seed: int = 42,
        percentage_of_layers_to_store: float = 1.0,
    ) -> list:
        """
        Generate and save attention maps.

        Args:
            prompt_generation (str): The text prompt for generation.
            num_samples (int): The number of samples to generate.
            sampler_params (dict): Additional parameters for the sampler.
            seed (int): The random seed.
            percentage_of_layers_to_store (float): The percentage of layers to store.

        Returns:
            dict: The generated attention maps.
                attn (torch.Tensor): The attention maps.
                output (dict): The generated output.
                coords (torch.Tensor): The coordinates of the generated sparse structure.
        """
        torch.manual_seed(seed)
        attn_globals.ATTN_COLLECT.set_store_attn(True)
        attn_globals.ATTN_COLLECT.set_percentage_of_layers_to_store(percentage_of_layers_to_store)
        cond = self.get_cond([prompt_generation])
        attn_cond = self.get_cond([prompt_attn])
        formats = ['mesh','gaussian']
        empty_cond = self.get_cond(['blank'])
        flow_model = self.models['sparse_structure_flow_model']
        reso = flow_model.resolution
        
        #generate the sparse structure
        noise = torch.randn(num_samples, flow_model.in_channels, reso, reso, reso).to(self.device)
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        z_s_generation = self.sparse_structure_sampler.sample(
            flow_model, noise, **cond, **sampler_params, verbose=True
        ).samples

        # reset the attention collector
        attn_globals.ATTN_COLLECT.set_store_attn(False)
        attn_globals.ATTN_COLLECT.set_percentage_of_layers_to_store(1.0)
        print(f" number of layers: {attn_globals.ATTN_COLLECT.get_num_of_layers()}")

        # sample SLAT with the empty condition
        decoder = self.models['sparse_structure_decoder']
        coords = torch.argwhere(decoder(z_s_generation)>0)[:, [0, 2, 3, 4]].int()
        slat = self.sample_slat(empty_cond, coords, {})
        output = self.decode_slat(slat, formats)
        
        return {"attn": attn_globals.ATTN_COLLECT.get(), "output": output, "coords": coords}

    
    def show_attn(self, attn: torch.Tensor, threshold: float = 0.5, original_coords: torch.Tensor = None, prompt_for_attn_slat: str = "blank", add_strength: bool = False):
        """
        Show the attention maps.

        Args:
            attn (torch.Tensor): The attention maps.
            threshold (float): The threshold for the attention map values.
            original_coords (torch.Tensor): The coordinates of the original sparse structure. [L, 4] columns: [batch, x, y, z]
            prompt_for_attn_slat (str): The text prompt for the attention SLAT.
            add_strength (bool): Whether to add the strength to the coordinates. [L, 5] columns: [batch, x, y, z, strength]

        Returns:
            dict: The generated output.
                output (dict): The generated output.
                coords (torch.Tensor): The coordinates of the generated sparse structure.
        """
        idx = original_coords[:, 1:4].long().to(attn.device)          # shape: (L, 3)
        empty_cond = self.get_cond([prompt_for_attn_slat])
        formats = ['mesh','gaussian']
        # (optional) validate bounds to avoid indexing errors
        if torch.any((idx < 0) | (idx >= 64)):
            raise ValueError("Some indices in attn[:,1:4] are out of [0, 63].")

        # Gather values from A at those coordinates
        vals = attn[idx[:, 0], idx[:, 1], idx[:, 2]]    # shape: (L,)

        # Build mask and filter rows
        mask = vals > threshold                              # shape: (L,)
        new_coords = original_coords[mask]
        if add_strength:
            # For each row in new_coords, get attn at [x, y, z] where x=new_coords[i,1], y=new_coords[i,2], z=new_coords[i,3]
            # new_coords shape: (N, 4), columns: [batch, x, y, z]
            x = new_coords[:, 1].cpu()
            y = new_coords[:, 2].cpu()
            z = new_coords[:, 3].cpu()
            # Gather strengths
            strengths = attn[x, y, z]
            # Convert back to tensor on same device as new_coords
            strengths = torch.from_numpy(strengths).to(new_coords.device)
            # Add as a new column
            new_coords_with_strength = torch.cat([new_coords, strengths.unsqueeze(1)], dim=1)
        slat = self.sample_slat(empty_cond, new_coords, {})
        return self.decode_slat(slat, formats), new_coords_with_strength
        
    def sample_sparse_structure(
        self,
        cond: dict,
        num_samples: int = 1,
        sampler_params: dict = {},
    ) -> torch.Tensor:
        """
        Sample sparse structures with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            num_samples (int): The number of samples to generate.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample occupancy latent
        flow_model = self.models['sparse_structure_flow_model']
        reso = flow_model.resolution
        if attn_globals.ATTN_COLLECT.get_noise() is not None:
            noise = attn_globals.ATTN_COLLECT.get_noise()
        else:
            noise = torch.randn(num_samples, flow_model.in_channels, reso, reso, reso).to(self.device)
            attn_globals.ATTN_COLLECT.set_noise(noise.clone())
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        z_s = self.sparse_structure_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True
        ).samples 
        # Decode occupancy latent
        decoder = self.models['sparse_structure_decoder']
        coords = torch.argwhere(decoder(z_s)>0)[:, [0, 2, 3, 4]].int()

        return coords

    def decode_slat(
        self,
        slat: sp.SparseTensor,
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
    ) -> dict:
        """
        Decode the structured latent.

        Args:
            slat (sp.SparseTensor): The structured latent.
            formats (List[str]): The formats to decode the structured latent to.

        Returns:
            dict: The decoded structured latent.
        """
        ret = {}
        if 'mesh' in formats:
            ret['mesh'] = self.models['slat_decoder_mesh'](slat)
        if 'gaussian' in formats:
            ret['gaussian'] = self.models['slat_decoder_gs'](slat)
        if 'radiance_field' in formats:
            ret['radiance_field'] = self.models['slat_decoder_rf'](slat)
        return ret
    
    def sample_slat(
        self,
        cond: dict,
        coords: torch.Tensor,
        sampler_params: dict = {},
    ) -> sp.SparseTensor:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            coords (torch.Tensor): The coordinates of the sparse structure.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample structured latent
        flow_model = self.models['slat_flow_model']
        noise = sp.SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = {**self.slat_sampler_params, **sampler_params}
        slat = self.slat_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True
        ).samples

        std = torch.tensor(self.slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        return slat

    @torch.no_grad()
    def run(
        self,
        prompt: str,
        prompt_old: str = None,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
    ) -> dict:
        """
        Run the pipeline.

        Args:
            prompt (str): The text prompt.
            num_samples (int): The number of samples to generate.
            seed (int): The random seed.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            formats (List[str]): The formats to decode the structured latent to.
        """
        cond = self.get_cond([prompt])
        cond_old = None
        if prompt_old is not None:
            cond_old = self.get_cond([prompt_old])
        torch.manual_seed(seed)
        coords = self.sample_sparse_structure(cond, num_samples, sparse_structure_sampler_params)
        slat = self.sample_slat(cond, coords, slat_sampler_params)
        return self.decode_slat(slat, formats)
    
    def voxelize(self, mesh: o3d.geometry.TriangleMesh) -> torch.Tensor:
        """
        Voxelize a mesh.

        Args:
            mesh (o3d.geometry.TriangleMesh): The mesh to voxelize.
            sha256 (str): The SHA256 hash of the mesh.
            output_dir (str): The output directory.
        """
        vertices = np.asarray(mesh.vertices)
        aabb = np.stack([vertices.min(0), vertices.max(0)])
        center = (aabb[0] + aabb[1]) / 2
        scale = (aabb[1] - aabb[0]).max()
        vertices = (vertices - center) / scale
        vertices = np.clip(vertices, -0.5 + 1e-6, 0.5 - 1e-6)
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(mesh, voxel_size=1/64, min_bound=(-0.5, -0.5, -0.5), max_bound=(0.5, 0.5, 0.5))
        vertices = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()])
        return torch.tensor(vertices).int().cuda()

    @torch.no_grad()
    def run_variant(
        self,
        mesh: o3d.geometry.TriangleMesh,
        prompt: str,
        num_samples: int = 1,
        seed: int = 42,
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
    ) -> dict:
        """
        Run the pipeline for making variants of an asset.

        Args:
            mesh (o3d.geometry.TriangleMesh): The base mesh.
            prompt (str): The text prompt.
            num_samples (int): The number of samples to generate.
            seed (int): The random seed
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            formats (List[str]): The formats to decode the structured latent to.
        """
        cond = self.get_cond([prompt])
        coords = self.voxelize(mesh)
        coords = torch.cat([
            torch.arange(num_samples).repeat_interleave(coords.shape[0], 0)[:, None].int().cuda(),
            coords.repeat(num_samples, 1)
        ], 1)
        torch.manual_seed(seed)
        slat = self.sample_slat(cond, coords, slat_sampler_params)
        return self.decode_slat(slat, formats)
