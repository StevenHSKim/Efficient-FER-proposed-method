1. (cmd) CSE-GResNet 폴더 접속
```
cd /userHome/userhome1/kimhaesung/FER_Models/Efficient-FER-proposed-method/CSE-GResNet
```

2. 도커 빌드 및 실행
```
docker compose build
docker compose up -d
```

3. 도커 bash 접속
```
docker compose exec -it cse_gresnet_env bash
```

4. 컨테이너 내부에서 현재 경로 확인 (/workspace여야 함)
```
pwd
ls -la
```

5. 실행
```
python src/main.py --dataset rafdb --iterations 10 --gpu 0
```