@echo off
chcp 65001 >nul
echo ========================================
echo  PaperPilot 文献索引服务 启动脚本 (OSS)
echo ========================================
echo.

where conda >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未检测到 conda，请先安装 Anaconda/Miniconda
    pause
    exit /b 1
)

call conda activate mildoc
if %errorlevel% neq 0 (
    echo [错误] conda 环境 mildoc 不存在
    pause
    exit /b 1
)

cd /d "%~dp0"

echo [info] 启动 mildoc_index 服务（阿里云 OSS 模式）...
echo [info] 日志文件：mildoc_index_oss.log
echo.

start "mildoc_index_oss" cmd /c "python main.py --provider oss --mode listen >> mildoc_index_oss.log 2>&1"

echo [ok] 服务已在后台启动
pause