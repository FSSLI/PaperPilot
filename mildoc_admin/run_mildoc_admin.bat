@echo off
chcp 65001 >nul
echo ========================================
echo  PaperPilot 文献管理后台 启动脚本
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

REM 安装 gunicorn（仅首次需要）
REM pip install gunicorn

echo [info] 启动 mildoc_admin 服务（端口 8870）...
echo [info] 日志文件：mildoc_admin.log
echo.

start "mildoc_admin" cmd /c "python -m gunicorn --workers 1 --bind 0.0.0.0:8870 admin_app:app >> mildoc_admin.log 2>&1"

echo [ok] 服务已在后台启动
echo [ok] 访问地址：http://127.0.0.1:8870
pause