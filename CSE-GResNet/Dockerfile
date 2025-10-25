# PyTorch 공식 이미지 사용 (GPU 지원)
FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel

# 환경 변수 설정
ENV DEBIAN_FRONTEND=noninteractive

# 기본 패키지 설치
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    vim \
    build-essential \
    libssl-dev \
    libffi-dev \
    libjpeg-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# PyTorch 1.5.1로 다운그레이드
RUN pip uninstall -y torch torchvision
RUN pip install torch==1.5.1 torchvision==0.6.1

# 기타 필요한 패키지들 설치
RUN pip install \
    numpy==1.19.5 \
    pandas==1.3.5 \
    scikit-learn==0.24.2 \
    matplotlib==3.3.4 \
    seaborn==0.11.2 \
    Pillow==8.4.0 \
    tqdm==4.64.1 \
    opencv-python==4.5.5.64 \
    thop

# 작업 디렉토리 생성
WORKDIR /workspace

# --- 추가된 내용 ---
COPY . .
RUN pip install ./models/Gabor_CNN_PyTorch/
# -----------------

# 컨테이너 시작 시 bash 실행
CMD ["/bin/bash"]