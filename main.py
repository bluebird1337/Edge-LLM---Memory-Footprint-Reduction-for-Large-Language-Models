from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import AnyStr
from llama_cpp import Llama

model_path = "llama2-oasst1-1k_merged.gguf"
llm = Llama(model_path=model_path,
           n_gpu_layers=-1 )

app = FastAPI()

class InferenceInput(BaseModel):
    prompt: AnyStr

def generate(prompt):
    response = llm(prompt, max_tokens=150)
    return response['choices'][0]['text']

@app.post("/predict/")
def predict(inference_input: InferenceInput):
    try:
        prediction = generate(inference_input.prompt)
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=7331)

# uvicorn main:app --reload --port 7331
