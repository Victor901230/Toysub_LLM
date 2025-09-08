import requests

class ToysubClient:
    def __init__(self,api_url:str):
        self.endpoint = api_url.rstrip("/") + "/api/chat"

    def ask(self,message:str)->dict:
        try:
            response = requests.post(self.endpoint,json={"message":message})    
            response.raise_for_status()
            result = response.json().get("reply",{})
            return {
                "text": result.get("text",""),
                "image": result.get("image", "")
            }
        except Exception as e:
            return {"text": f"發生錯誤：{e}", "image": ""}