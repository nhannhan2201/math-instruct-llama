import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.inference import MathSolver

# Khởi tạo FastAPI app
app = FastAPI(title="Math Instruct Llama API", description="API giải toán bằng mô hình Llama QLoRA")

# Load model (Khởi tạo ở ngoài để model chỉ cần load 1 lần vào RAM/VRAM khi start server)
try:
    solver = MathSolver(method="qlora")
except Exception as e:
    print(f"Lỗi khi load model: {e}")
    solver = None

# Cấu trúc dữ liệu đầu vào Nhan mong muốn
class MathRequest(BaseModel):
    question: str
    max_new_tokens: int = 150
    temperature: float = 0.7

@app.post("/solve")
def solve_math(req: MathRequest):
    if solver is None:
        raise HTTPException(status_code=500, detail="Model chưa được load thành công.")
    
    try:
        # Gọi hàm solve từ class MathSolver của Nhan
        answer = solver.solve(
            question=req.question, 
            max_new_tokens=req.max_new_tokens, 
            temperature=req.temperature
        )
        return {
            "success": True, 
            "question": req.question, 
            "answer": answer
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def health_check():
    return {"status": "OK", "message": "Math LLM Server is running!"}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)