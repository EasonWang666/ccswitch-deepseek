@echo off
chcp 65001 >nul
echo ==============================================
echo          正在启动 DeepSeek 代理服务...
echo ==============================================
echo.

:: 执行 npm start
npm start

:: 防止窗口闪退
pause