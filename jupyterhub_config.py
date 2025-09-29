"""
🔬 JupyterHub Configuration for QFrame Research Platform
"""

import os
import sys

# JupyterHub configuration
c = get_config()  # noqa

# Basic configuration
c.JupyterHub.ip = '0.0.0.0'
c.JupyterHub.port = 8000
c.JupyterHub.hub_ip = '0.0.0.0'

# Authentication - Simple for development, replace with OAuth for production
c.JupyterHub.authenticator_class = 'jupyterhub.auth.DummyAuthenticator'
c.DummyAuthenticator.password = os.environ.get('JUPYTERHUB_PASSWORD', 'qframe123')

# Admin users
c.Authenticator.admin_users = {'admin', 'qframe'}
c.Authenticator.allowed_users = {'admin', 'qframe', 'researcher1', 'researcher2', 'researcher3'}

# Spawner configuration - DockerSpawner for isolated environments
c.JupyterHub.spawner_class = 'dockerspawner.DockerSpawner'
c.DockerSpawner.image = os.environ.get('DOCKER_NOTEBOOK_IMAGE', 'qframe/research-notebook:latest')
c.DockerSpawner.network_name = 'qframe-research'
c.DockerSpawner.remove = True
c.DockerSpawner.debug = True

# Notebook directory and volume mounts
notebook_dir = '/home/jovyan'
c.DockerSpawner.notebook_dir = notebook_dir
c.DockerSpawner.volumes = {
    'jupyterhub-user-{username}': notebook_dir,
    '/home/jim/DEV/claude-code/quant_framework_research/data': {
        'bind': '/home/jovyan/data',
        'mode': 'ro'  # Read-only access to shared data
    },
    '/home/jim/DEV/claude-code/quant_framework_research/qframe': {
        'bind': '/home/jovyan/qframe',
        'mode': 'ro'  # Read-only access to qframe code
    }
}

# Environment variables for spawned containers
c.DockerSpawner.environment = {
    'JUPYTER_ENABLE_LAB': 'yes',
    'MLFLOW_TRACKING_URI': 'http://mlflow:5000',
    'DASK_SCHEDULER': 'tcp://dask-scheduler:8786',
    'RAY_ADDRESS': 'ray://ray-head:10001',
    'TIMESCALEDB_URL': 'postgresql://qframe:qframe123@timescaledb:5432/qframe_research',
    'MINIO_ENDPOINT': 'http://minio:9000',
    'MINIO_ACCESS_KEY': os.environ.get('MINIO_ROOT_USER', 'minio'),
    'MINIO_SECRET_KEY': os.environ.get('MINIO_ROOT_PASSWORD', 'minio123'),
}

# Resource limits per user
c.DockerSpawner.mem_limit = '8G'
c.DockerSpawner.cpu_limit = 4
c.DockerSpawner.mem_guarantee = '2G'
c.DockerSpawner.cpu_guarantee = 1

# Persistent storage
c.DockerSpawner.volumes = {
    'jupyterhub-user-{username}': {
        'bind': notebook_dir,
        'mode': 'rw'
    },
    'qframe-shared-data': {
        'bind': '/home/jovyan/shared',
        'mode': 'ro'
    },
    'qframe-models': {
        'bind': '/home/jovyan/models',
        'mode': 'rw'
    }
}

# User options for different notebook configurations
c.DockerSpawner.options_form = """
<label for="stack">Select your notebook stack:</label>
<select name="stack" size="1">
<option value="qframe/research-notebook:latest">QFrame Research (Full Stack)</option>
<option value="qframe/research-notebook:ml">QFrame ML (PyTorch/TensorFlow)</option>
<option value="qframe/research-notebook:quant">QFrame Quant (Financial Libraries)</option>
<option value="qframe/research-notebook:minimal">QFrame Minimal (Core Only)</option>
</select>

<label for="cpu_limit">CPU Cores:</label>
<select name="cpu_limit" size="1">
<option value="1">1 Core</option>
<option value="2" selected>2 Cores</option>
<option value="4">4 Cores</option>
<option value="8">8 Cores</option>
</select>

<label for="mem_limit">Memory:</label>
<select name="mem_limit" size="1">
<option value="2G">2 GB</option>
<option value="4G" selected>4 GB</option>
<option value="8G">8 GB</option>
<option value="16G">16 GB</option>
</select>

<label for="gpu">Enable GPU:</label>
<select name="gpu" size="1">
<option value="no" selected>No</option>
<option value="yes">Yes (NVIDIA)</option>
</select>
"""

def options_form_parser(formdata):
    """Parse the options form"""
    options = {}

    options['image'] = formdata.get('stack', ['qframe/research-notebook:latest'])[0]
    options['cpu_limit'] = float(formdata.get('cpu_limit', ['2'])[0])
    options['mem_limit'] = formdata.get('mem_limit', ['4G'])[0]

    # GPU support
    if formdata.get('gpu', ['no'])[0] == 'yes':
        options['extra_host_config'] = {
            'runtime': 'nvidia',
            'device_requests': [
                {
                    'driver': 'nvidia',
                    'count': 1,
                    'capabilities': [['gpu']]
                }
            ]
        }

    return options

c.DockerSpawner.options_form = options_form
c.DockerSpawner.options_fn = options_form_parser

# Services - Additional services available to notebooks
c.JupyterHub.services = [
    {
        'name': 'idle-culler',
        'admin': True,
        'command': [
            sys.executable,
            '-m', 'jupyterhub_idle_culler',
            '--timeout=3600',  # 1 hour idle timeout
            '--max-age=86400',  # 24 hour max age
        ],
    },
    {
        'name': 'mlflow',
        'url': 'http://mlflow:5000',
        'display': True,
    },
    {
        'name': 'dask-dashboard',
        'url': 'http://dask-scheduler:8787',
        'display': True,
    },
    {
        'name': 'ray-dashboard',
        'url': 'http://ray-head:8265',
        'display': True,
    },
    {
        'name': 'optuna-dashboard',
        'url': 'http://optuna-dashboard:8080',
        'display': True,
    },
    {
        'name': 'superset',
        'url': 'http://superset:8088',
        'display': True,
    }
]

# Database for JupyterHub state
c.JupyterHub.db_url = 'postgresql://jupyterhub:jupyterhub@postgres-jupyterhub:5432/jupyterhub'

# Cleanup settings
c.JupyterHub.cleanup_servers = True
c.JupyterHub.cleanup_proxy = True

# SSL/TLS Configuration (for production)
# c.JupyterHub.ssl_cert = '/etc/jupyterhub/ssl/cert.pem'
# c.JupyterHub.ssl_key = '/etc/jupyterhub/ssl/key.pem'

# Logging
c.JupyterHub.log_level = 'INFO'
c.DockerSpawner.debug = False

# Template paths for custom UI
c.JupyterHub.template_paths = ['/srv/jupyterhub/templates']

# Logo and branding
c.JupyterHub.logo_file = '/srv/jupyterhub/static/qframe-logo.png'

# Custom pre-spawn hook
def pre_spawn_hook(spawner):
    """Hook called before spawning a notebook"""
    username = spawner.user.name
    spawner.log.info(f"Pre-spawn hook for user: {username}")

    # Create user-specific MLflow experiment
    spawner.environment['MLFLOW_EXPERIMENT_NAME'] = f"/Users/{username}/experiments"

    # Set user-specific S3 bucket path
    spawner.environment['S3_USER_PATH'] = f"s3://qframe-research/{username}/"

c.DockerSpawner.pre_spawn_hook = pre_spawn_hook

# Post-spawn hook
def post_spawn_hook(spawner):
    """Hook called after spawning a notebook"""
    username = spawner.user.name
    spawner.log.info(f"Notebook spawned successfully for user: {username}")

c.DockerSpawner.post_spawn_hook = post_spawn_hook

# Shutdown settings
c.JupyterHub.shutdown_on_logout = False
c.DockerSpawner.remove = True

print("✅ QFrame JupyterHub configuration loaded successfully!")
print("🔬 Research platform ready for multi-user access")
print("📊 Integrated services: MLflow, Dask, Ray, Optuna, Superset")