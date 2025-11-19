import requests
from ...utils import RemoteOpenAIServer

MODEL_NAME = "meta-llama/Llama-3.2-1B"

def test_admin_auth_env():
    # Test authentication with VLLM_ADMIN_API_KEY env var
    args = [
        "--dtype", "bfloat16",
        "--max-model-len", "8192",
        "--max-num-seqs", "128",
        "--enable-sleep-mode",
    ]
    
    admin_key = "secret_admin_key"
    env_dict = {
        "VLLM_ADMIN_API_KEY": admin_key,
        "CUDA_VISIBLE_DEVICES": "0",
    }

    with RemoteOpenAIServer(MODEL_NAME, args, env_dict=env_dict) as remote_server:
        # Test unauthorized access
        response = requests.get(remote_server.url_for("is_sleeping"))
        assert response.status_code == 401

        # Test authorized access
        headers = {"Authorization": f"Bearer {admin_key}"}
        response = requests.get(remote_server.url_for("is_sleeping"), headers=headers)
        assert response.status_code == 200

        # Test wrong key
        headers = {"Authorization": "Bearer wrong_key"}
        response = requests.get(remote_server.url_for("is_sleeping"), headers=headers)
        assert response.status_code == 401

        # Test standard endpoints (should be open as no standard api key set)
        response = requests.get(remote_server.url_for("health"))
        assert response.status_code == 200

def test_admin_auth_cli():
    # Test authentication with --admin-api-key CLI arg
    admin_key = "secret_admin_cli_key"
    args = [
        "--dtype", "bfloat16",
        "--max-model-len", "8192",
        "--max-num-seqs", "128",
        "--enable-sleep-mode",
        "--admin-api-key", admin_key,
    ]
    
    env_dict = {
        "CUDA_VISIBLE_DEVICES": "0",
    }

    with RemoteOpenAIServer(MODEL_NAME, args, env_dict=env_dict) as remote_server:
        # Test unauthorized access
        response = requests.get(remote_server.url_for("is_sleeping"))
        assert response.status_code == 401

        # Test authorized access
        headers = {"Authorization": f"Bearer {admin_key}"}
        response = requests.get(remote_server.url_for("is_sleeping"), headers=headers)
        assert response.status_code == 200

def test_separate_auth():
    # Test with both standard API key and Admin API key
    admin_key = "admin_key"
    user_key = "user_key"
    
    args = [
        "--dtype", "bfloat16",
        "--max-model-len", "8192",
        "--max-num-seqs", "128",
        "--enable-sleep-mode",
        "--api-key", user_key,
        "--admin-api-key", admin_key,
    ]
    
    env_dict = {
        "CUDA_VISIBLE_DEVICES": "0",
    }

    with RemoteOpenAIServer(MODEL_NAME, args, env_dict=env_dict) as remote_server:
        # User key should work for standard endpoints (mock check, e.g. /v1/models)
        headers_user = {"Authorization": f"Bearer {user_key}"}
        response = requests.get(remote_server.url_for("v1/models"), headers=headers_user)
        assert response.status_code == 200

        # User key should NOT work for admin endpoints
        response = requests.get(remote_server.url_for("is_sleeping"), headers=headers_user)
        assert response.status_code == 401

        # Admin key should work for admin endpoints
        headers_admin = {"Authorization": f"Bearer {admin_key}"}
        response = requests.get(remote_server.url_for("is_sleeping"), headers=headers_admin)
        assert response.status_code == 200
        
        # Admin key should NOT work for standard endpoints (unless same key used, which is not the case here)
        response = requests.get(remote_server.url_for("v1/models"), headers=headers_admin)
        assert response.status_code == 401

