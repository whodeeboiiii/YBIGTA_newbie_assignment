
# anaconda(또는 miniconda)가 존재하지 않을 경우 설치해주세요!
if ! command -v conda &> /dev/null; then
    echo "[INFO] Conda not found. Installing Miniconda..."
    
    OS=$(uname -s)
    ARCH=$(uname -m)
    
    if [[ "$OS" == "Darwin" ]]; then
        if [[ "$ARCH" == "x86_64" ]]; then
            MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh"
        elif [[ "$ARCH" == "arm64" ]]; then
            MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh"
        else
            echo "[ERROR] Unsupported architecture: $ARCH"
            exit 1
        fi
    else
        echo "[ERROR] This script is designed for macOS."
        exit 1
    fi
    
    curl -o miniconda.sh "$MINICONDA_URL"
    bash miniconda.sh -b -p "$HOME/miniconda"
    rm miniconda.sh
    
    source "$HOME/miniconda/etc/profile.d/conda.sh"
    echo "[INFO] Miniconda installed and activated."
else
    CONDA_BASE=$(conda info --base)
    if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
        source "$CONDA_BASE/etc/profile.d/conda.sh"
    else
        echo "[ERROR] Could not find conda.sh at $CONDA_BASE"
        exit 1
    fi
fi


# Conda 환셩 생성 및 활성화
if conda info --envs | grep -q "myenv"; then
    echo "[INFO] Environment 'myenv' exists."
else
    echo "[INFO] Creating environment 'myenv'..."
    conda create -n myenv python=3.10 -y
fi

conda activate myenv

## 건드리지 마세요! ##
python_env=$(python -c "import sys; print(sys.prefix)")
if [[ "$python_env" == *"/envs/myenv"* ]]; then
    echo "[INFO] 가상환경 활성화: 성공"
else
    echo "[INFO] 가상환경 활성화: 실패"
    exit 1 
fi

# 필요한 패키지 설치
conda install mypy -y

# Submission 폴더 파일 실행
cd submission || { echo "[INFO] submission 디렉토리로 이동 실패"; exit 1; }

for file in *.py; do
    file_name=$(basename "$file" .py)
    problem_num="${file_name##*_}"
    input_file="../input/${problem_num}_input"
    output_file="../output/${problem_num}_output"
    
    if [[ -f "$input_file" ]]; then
        echo "[INFO] Running $file with input $input_file..."
        python "$file" < "$input_file" > "$output_file"
    else
        echo "[WARN] Input file for $file not found."
    fi

done

# mypy 테스트 실행 및 mypy_log.txt 저장
mypy . > ../mypy_log.txt || true

# conda.yml 파일 생성
conda env export > ../conda.yml

# 가상환경 비활성화
conda deactivate