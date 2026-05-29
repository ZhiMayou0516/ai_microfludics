# LAMP 曲线智能判读：微信小程序版

这个包已经把你原来的 Python 判读逻辑保留下来，并新增了两个部分：

- `backend/`：FastAPI 后端，负责读取 CSV/XLSX、曲线处理、规则判读和 AI 判读。
- `miniprogram/`：微信小程序前端，负责上传文件、显示总结果、查看曲线和孔位详情。

原来的算法核心在 `src/lamp_ai/`，模型文件在 `models/lamp_v7_well_ai.joblib`。没有把临床建库输出大文件打包进来，所以项目体积会小很多。

## 1. 先启动 Python 后端

进入本目录后运行：

```bash
python -m pip install -r backend/requirements.txt
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Windows 也可以直接双击：

```text
run_backend.bat
```


## 1.1 AI 模型版本提醒

当前打包的 `models/lamp_v7_well_ai.joblib` 是按原项目环境保存的，建议后端环境使用：

```bash
pip install scikit-learn==1.6.1
```

如果只用“规则判读”，通常不受这个模型版本影响。

启动成功后，浏览器打开：

```text
http://127.0.0.1:8000/api/health
```

看到 `ok: true` 就说明后端正常。

## 2. 用微信开发者工具打开小程序

微信开发者工具里选择：

```text
导入项目 → 选择 miniprogram 文件夹
```

`appid` 现在写的是 `touristappid`，可以先用测试号或自己的 AppID 替换。

开发者工具本机调试时，默认后端地址是：

```text
http://127.0.0.1:8000
```

配置位置：

```text
miniprogram/utils/config.js
```

## 3. 真机预览怎么改

真机上 `127.0.0.1` 指的是手机自己，不是电脑。所以真机预览时，需要把 `config.js` 里的地址改成电脑局域网 IP，例如：

```js
const BASE_URL = 'http://192.168.1.23:8000'
```

电脑 IP 可以在 Windows PowerShell 里查：

```bash
ipconfig
```

找无线网卡的 IPv4 地址。手机和电脑必须连接同一个 WiFi。

## 4. 正式发布注意

微信小程序正式发布时，后端必须满足：

- 使用 HTTPS；
- 有备案/可访问域名；
- 在微信公众平台配置 `request` 和 `uploadFile` 合法域名；
- 不能只用局域网 IP。

比赛展示阶段可以先用开发者工具或真机调试模式。

## 5. 文件格式

当前后端沿用你原来的读取逻辑：

- 每一列对应一个孔位；
- 每一行对应一个时间点；
- 支持 CSV、XLSX、XLS；
- 默认去除前 5 行；
- 如果第一列是 time/序号，可以在小程序高级设置里打开“第一列为 time”。

## 6. 当前页面结构

小程序目前有 3 个页面：

```text
pages/index   上传文件 + 选择规则/AI + 开始检测
pages/result  总结果 + 阳性/阴性/需复核/异常数量
pages/detail  曲线图 + 单孔详情 + 全部孔位列表
```

## 7. 不要动这些核心文件

如果只是改界面，优先改：

```text
miniprogram/pages/**/*.wxml
miniprogram/pages/**/*.wxss
```

暂时不要改：

```text
src/lamp_ai/rules.py
src/lamp_ai/features.py
src/lamp_ai/v7_ai.py
```

这些是你的判读逻辑和模型相关代码。
