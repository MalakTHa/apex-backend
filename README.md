# APEX Backend

Backend service for the APEX (Algorithm Performance & Efficiency X-ray) platform.

The backend is responsible for analyzing Python source code, estimating algorithm complexity, generating optimization suggestions, applying safe code optimizations, and simulating algorithm performance.

## Features

- Python syntax validation
- Function detection
- Function-level analysis
- Time complexity estimation
- Space complexity estimation
- Algorithm pattern detection
- Performance warnings
- Optimization suggestions
- Simple Optimization
- Algorithm Replacement
- Safe optimization layer
- Performance simulation
- REST API using FastAPI

## Technologies

- Python
- FastAPI
- Uvicorn
- Pydantic
- AST (Abstract Syntax Tree)

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd backend
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it.

Windows:

```bash
.venv\Scripts\activate
```

Linux / macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install fastapi uvicorn
```

## Run the server

```bash
uvicorn main:app --reload
```

The API will be available at:

```
https://apex-backend-if96.onrender.com
```

Swagger documentation:

```
https://apex-backend-if96.onrender.com/docs
```

## API Endpoints

### Analyze Code

```
POST /analyze
```

Analyzes the selected Python function and returns:

- Time Complexity
- Space Complexity
- Algorithm Type
- Loop Information
- Warnings
- Suggestions

---

### Simple Optimization

```
POST /optimize/simple
```

Applies safe optimizations without changing the underlying algorithm.

---

### Algorithm Replacement

```
POST /optimize/replacement
```

Replaces inefficient implementations with more efficient algorithms or data structures when a trusted optimization pattern is detected.

---

### Simulation

```
POST /simulate
```

Generates performance comparison data between the original and optimized implementations.


## Notes

- Source code is analyzed statically using Python AST.
- The backend never executes the user's Python code.
- Optimizations are applied only when trusted patterns are detected to preserve program behavior.