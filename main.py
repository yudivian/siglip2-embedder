#!/usr/bin/env python3

"""
Microservice and CLI with FastAPI to generate text/image embeddings with SigLIP 2.
"""

import argparse
import logging
import os
import sys
import io
import torch
import numpy as np
import yaml
from PIL import Image
from transformers import AutoProcessor, Siglip2Model
import json
import pickle

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)

class Siglip2Embedder:
    """
    Class to generate text and image embeddings using SigLIP 2.
    """
    def __init__(self, model_name: str, device_choice: str):
        self.model_name = model_name
        self.device, self.dtype = self._setup_device(device_choice)
        logging.info(f"Final device in use: {self.device}")

        try:
            logging.info(f"Loading model '{self.model_name}'...")
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = Siglip2Model.from_pretrained(
                self.model_name,
                torch_dtype=self.dtype
            ).to(self.device)
            self.model.eval()
            logging.info("Model loaded successfully ✅")
        except Exception as e:
            logging.error(f"Error loading the model: {e}")
            raise

    def _setup_device(self, device_choice: str):
        if device_choice == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            device = device_choice
        
        if device == 'cuda' and torch.cuda.is_available():
            logging.info("Using GPU (CUDA) with float16")
            return torch.device("cuda"), torch.float16
        logging.info("Using CPU with float32")
        return torch.device("cpu"), torch.float32

    def _normalize_embedding(self, features: torch.Tensor) -> np.ndarray:
        features = features / features.norm(p=2, dim=-1, keepdim=True)
        return features.cpu().detach().numpy().astype(np.float32)

    def get_text_embedding(self, text: str) -> np.ndarray:
        logging.info(f"Processing text: '{text[:50]}...'")
        with torch.no_grad():
            inputs = self.processor(text=[text], return_tensors="pt", padding="max_length", max_length=64, truncation=True).to(self.device)
            text_features = self.model.get_text_features(**inputs)
            return self._normalize_embedding(text_features)

    def get_image_embedding(self, image_path: str) -> np.ndarray:
        logging.info(f"Processing image from path: {image_path}...")
        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            logging.error(f"Image file not found at: {image_path}")
            raise
        except Exception as e:
            logging.error(f"Error processing image from path: {e}")
            raise

        with torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            image_features = self.model.get_image_features(**inputs)
            return self._normalize_embedding(image_features)
            
    def get_image_embedding_from_bytes(self, image_bytes: bytes) -> np.ndarray:
        logging.info("Processing image from bytes...")
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            logging.error(f"Error while processing the image from bytes: {e}")
            raise ValueError("The provided image file is invalid.")

        with torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            image_features = self.model.get_image_features(**inputs)
            return self._normalize_embedding(image_features)


def load_configuration(custom_config_path=None):
    """Load configuration from .yaml files"""
    
    config = {
        'device': 'auto',
        'model_name': 'google/siglip2-base-patch16-naflex',
        'server_host': '0.0.0.0',
        'server_port': 8000
    }
    
    default_config_file = 'config.yaml'
    if os.path.exists(default_config_file):
        try:
            with open(default_config_file, 'r') as f:
                yaml_config = yaml.safe_load(f)
                if yaml_config:
                    config.update(yaml_config)
                    logging.info(f"Default config file '{default_config_file}' found and loaded.")
        except Exception as e:
            logging.warning(f"Could not read '{default_config_file}', using defaults: {e}")

    if custom_config_path:
        if os.path.exists(custom_config_path):
            try:
                with open(custom_config_path, 'r') as f:
                    custom_yaml_config = yaml.safe_load(f)
                    if custom_yaml_config:
                        config.update(custom_yaml_config) 
                        logging.info(f"Custom config '{custom_config_path}' loaded successfully.")
            except Exception as e:
                logging.error(f"Error loading custom config '{custom_config_path}': {e}")
        else:
            logging.warning(f"Custom config file specified but not found: '{custom_config_path}'")
            
    return config

def run_cli(args, config):
    """Runs the command-line interface functionality."""
    embedder = Siglip2Embedder(
        model_name=config['model_name'],
        device_choice=config['device']
    )
    
    embedding = None
    if args.text:
        embedding = embedder.get_text_embedding(args.text)
    elif args.image:
        try:
            embedding = embedder.get_image_embedding(args.image)
        except FileNotFoundError:
            sys.exit(1) 
        except Exception as e:
            logging.error(f"Could not process the image: {e}")
            sys.exit(1)

    if embedding is not None:
        if args.output:
            logging.info(f"Saving embedding to '{args.output}' in {args.format} format.")
            if args.format == 'json':
                with open(args.output, 'w') as f:
                    json.dump(embedding.flatten().tolist(), f)
            elif args.format == 'pickle':
                with open(args.output, 'wb') as f:
                    pickle.dump(embedding, f)
            logging.info("File saved successfully.")
        else:
            print(embedding.flatten().tolist())
        logging.info(f"Embedding generated with shape: {embedding.shape}")


def run_server(config):
    """Starts the FastAPI server."""
    import uvicorn
    from fastapi import FastAPI, UploadFile, File, HTTPException
    from pydantic import BaseModel
    from contextlib import asynccontextmanager

    embedder_instance = None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal embedder_instance
        logging.info(f"Configuration loaded: Model='{config['model_name']}', Device='{config['device']}'")
        embedder_instance = Siglip2Embedder(
            model_name=config['model_name'],
            device_choice=config['device']
        )
        yield
        logging.info("Shutting down service.")

    app = FastAPI(
        title="SigLIP 2 Embedding API",
        description="A microservice to generate text/image embeddings with SigLIP 2",
        version="1.0.0",
        lifespan=lifespan
    )

    class TextRequest(BaseModel):
        text: str

    class EmbeddingResponse(BaseModel):
        embedding: list[float]
        model: str
        shape: tuple[int, int]

    @app.post("/embed/text", response_model=EmbeddingResponse)
    async def embed_text(request: TextRequest):
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        embedding_array = embedder_instance.get_text_embedding(request.text)
        
        return {
            "embedding": embedding_array.flatten().tolist(),
            "model": embedder_instance.model_name,
            "shape": embedding_array.shape
        }

    @app.post("/embed/image", response_model=EmbeddingResponse)
    async def embed_image(file: UploadFile = File(...)):
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File is not a valid image")

        try:
            image_bytes = await file.read()
            embedding_array = embedder_instance.get_image_embedding_from_bytes(image_bytes)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        return {
            "embedding": embedding_array.flatten().tolist(),
            "model": embedder_instance.model_name,
            "shape": embedding_array.shape
        }

    @app.get("/", include_in_schema=False)
    async def root():
        return {"message": "Welcome to the SigLIP 2 Embeddings API. See /docs for documentation."}

    uvicorn.run(app, host=config['server_host'], port=config['server_port'])


def main():
    """Main entry point that decides whether to run the CLI or the server."""
    parser = argparse.ArgumentParser(
        description="SigLIP 2 embedding generator. Works as a CLI or a web server."
    )
    
    parser.add_argument(
        '--mode', 
        choices=['cli', 'server'], 
        default='server', 
        help="Execution mode: 'cli' for the command line or 'server' to start the web server (default)."
    )
    parser.add_argument(
        '--text', 
        type=str, 
        help="Input text to generate an embedding (cli mode only)."
    )
    parser.add_argument(
        '--image', 
        type=str, 
        help="Path to the image file to generate an embedding (cli mode only)."
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help="Path to a custom YAML configuration file. Overrides 'config.yaml'."
    )
    
    parser.add_argument(
        '--device',
        type=str,
        help="Device to use for computation, e.g., 'cuda', 'cpu'. Overrides config file."
    )
    parser.add_argument(
        '--model-name',
        type=str,
        help="Name of the SigLIP2 model from Hugging Face. Overrides config file."
    )
    parser.add_argument(
        '--output',
        '-o',
        type=str,
        help="Path to the output file (json or pickle)."
    )
    parser.add_argument(
        '--format',
        type=str,
        choices=['json', 'pickle'],
        default='json',
        help="Output file format (json or pickle). Default is json."
    )
    parser.add_argument(
        '--host',
        type=str,
        help="Host for the server (server mode only)."
    )
    parser.add_argument(
        '--port',
        type=int,
        help="Port for the server (server mode only)."
    )

    args = parser.parse_args()
    
    if args.mode == 'cli' and not args.text and not args.image:
        parser.error("in cli mode, argument --text or --image is required.")
    
    if args.mode == 'cli' and args.text and args.image:
        parser.error("in cli mode, provide either --text or --image, not both.")

    config = load_configuration(custom_config_path=args.config)

    if args.model_name:
        logging.info(f"Overriding 'model_name' from config file with CLI argument: '{args.model_name}'")
        config['model_name'] = args.model_name
    
    if args.device:
        logging.info(f"Overriding 'device' from config file with CLI argument: '{args.device}'")
        config['device'] = args.device

    if args.host:
        config['server_host'] = args.host
    
    if args.port:
        config['server_port'] = args.port

    if args.mode == 'cli':
        run_cli(args, config)
    else:
        run_server(config)

if __name__ == "__main__":
    main()