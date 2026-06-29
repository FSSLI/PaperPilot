@echo off
chcp 65001 >nul
echo ========================================
echo  PaperPilot 文献索引服务 启动脚本
echo ========================================
echo.

REM 检查 conda 环境
where conda >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未检测到 conda，请先安装 Anaconda/Miniconda
    pause
    exit /b 1
)

REM 激活 conda 环境
call conda activate mildoc
if %errorlevel% neq 0 (
    echo [错误] conda 环境 mildoc 不存在，请先创建：
    echo        conda create -n mildoc python=3.12
    echo        conda activate mildoc
    echo        pip install -r requirements.txt
    pause
    exit /b 1
)

REM 切换到脚本所在目录
cd /d "%~dp0"

REM 安装依赖（仅首次或依赖变更时需要）
REM pip install -r requirements.txt

echo [info] 启动 mildoc_index 服务（MinIO 模式）...
echo [info] 日志文件：mildoc_index.log
echo.

REM 后台运行（调试时可改为前台的 python main.py）
start "mildoc_index" cmd /c "python main.py --provider minio --mode listen >> mildoc_index.log 2>&1"

echo [ok] 服务已在后台启动
echo [ok] 查看日志：type mildoc_index.log
echo [ok] 停止服务：在任务管理器中结束 python.exe 进程
pause