# StreamBooth

StreamBooth is a Streamlit wrapper to run DreamBooth for training Stable Diffusion models on local hardware.

## Installation

### Local

*Clone the repository:*

```bash
git clone https://github.com/willmo103/StreamBooth.git
cd StreamBooth
```

*Create a virtual environment:*

```bash
python3 -m venv venv

source venv/bin/activate
```

*Install the requirements:*

```bash
pip install -r requirements.txt
```

*Run the Streamlit app:*

```bash
streamlit run app.py
```

### Docker-Compose

*Clone the repository:*

```bash
git clone https://github.com/willmo103/StreamBooth.git
cd StreamBooth
```

*Build the Docker image:*

```bash
docker-compose build
```

## Usage

*Run the Streamlit app:*

```bash
streamlit run app.py

# or

docker-compose up
```
