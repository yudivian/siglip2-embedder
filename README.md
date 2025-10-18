# SigLIP 2 Embedder API and CLI

This project provides a microservice and a command-line interface (CLI) to generate text and image embeddings using Google's **SigLIP 2** models. It is built with FastAPI for the API and Hugging Face `transformers` for model inference.

## ✨ Features

  * **Dual Mode**: Can run as a REST API server or a command-line tool (CLI).
  * **Text & Image Embedding**: Generates normalized embeddings for text and image inputs.
  * **Flexible Output**: The CLI can print embeddings to standard output or save them to JSON or pickle files.
  * **Layered Configuration**: Manages settings via a default `config.yaml`, which can be overridden by a custom config file.
  * **Auto Hardware Detection**: Automatically detects and uses a GPU (CUDA) if available for faster performance.
  * **Documented API**: The API interface is automatically documented via Swagger UI (`/docs`) and ReDoc (`/redoc`).

-----

## Table of Contents

  * [Installation](https://www.google.com/search?q=%23-installation)
  * [Configuration](https://www.google.com/search?q=%23-configuration)
  * [Usage](https://www.google.com/search?q=%23-usage)
      * [Server (API) Mode](https://www.google.com/search?q=%23server-api-mode)
      * [CLI Mode](https://www.google.com/search?q=%23cli-mode)
  * [Project Structure](https://www.google.com/search?q=%23-project-structure)

-----

## 🚀 Installation

1.  **Clone the repository.**
2.  **Create a virtual environment (recommended).**
3.  **Install dependencies.** The project uses the dependencies listed in `pyproject.toml`.

-----

## ⚙️ Configuration

You can configure the application's behavior by editing the `config.yaml` file, which is loaded by default if it exists.

  * **`device`**: Defines the compute device. If set to `auto`, the system will use `cuda` if available; otherwise, it will fall back to `cpu`.
  * **`model_name`**: The identifier of the SigLIP 2 model to download from Hugging Face.
  * **`server_host`**: IP address for the server to bind to.
  * **`server_port`**: Port for the server to run on.

You can **override** these settings in two ways when running the application:

1.  **Custom Config File**: By passing the `--config` argument with a path to your own YAML file. Settings in this file will override `config.yaml`.
2.  **Direct Arguments**: By passing specific arguments like `--device` or `--model-name` (see CLI mode). These override all config files.

-----

## ▶️ Usage

The application can run in two modes: `server` (default) or `cli`.

### Server (API) Mode

This mode starts a FastAPI web server, exposing endpoints for generating embeddings.

By default, the server will be available at `http://0.0.0.0:8000`. You can change this with command-line arguments.

#### API Endpoints

  * `GET /`
      * **Description**: Displays a welcome message.
  * `POST /embed/text`
      * **Description**: Generates an embedding for a given text.
  * `POST /embed/image`
      * **Description**: Generates an embedding for an image.

Once the server is running, you can access the interactive documentation at:

  * **Swagger UI**: `http://127.0.0.1:8000/docs`
  * **ReDoc**: `http://127.0.0.1:8000/redoc`

### CLI Mode

This mode allows you to generate embeddings directly from your terminal without starting a web server.

The result will be printed to the console unless an `--output` file is specified.

#### Overriding Configuration from the CLI

You can change settings without editing the `config.yaml` file by using command-line arguments.

  * **To use a custom configuration file:**

    ```bash
    siglip2-embedder --mode cli --config my_custom_config.yaml --text "Hello world"
    ```

  * **To override specific settings:**

    ```bash
    siglip2-embedder --mode cli --model-name "google/siglip2-base-patch16-224" --device "cpu" --text "This runs on the CPU"
    ```

-----

## 📂 Project Structure

  * **`main.py`**: Contains the main application logic (CLI and API).
  * **`config.yaml`**: Default configuration file.
  * **`pyproject.toml`**: Project definition and dependencies.
  * **`README.md`**: This file.