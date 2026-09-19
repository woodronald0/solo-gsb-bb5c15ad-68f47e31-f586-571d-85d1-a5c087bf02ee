# 平面照片标尺校准 Web 应用

上传一张包含**已知实际宽 × 高矩形参照物**（信用卡、打印矩形、标定框等）的平面照片，
依次标记矩形四角后，系统通过**四点 DLT 单应矩阵**完成透视校准；随后可在照片上
绘制一个或多个闭合多边形，实时得到**校正后的实际面积与周长**。

- 后端：FastAPI（Python 3.11），NumPy 求解/校验 3×3 单应矩阵
- 前端：原生 HTML / CSS / JavaScript，Canvas 渲染，零构建、零前端框架
- 所有几何点以**原始图像像素坐标**存储，缩放 / 平移 / 拖动顶点不改变测量几何
- 前后端使用同一套数值算法（镜像实现），浏览器实时预览，服务器权威复核

## 一条命令启动

需要 Docker（含 Compose 插件）：

```bash
docker compose up -d --build
```

然后在浏览器打开：

> **http://localhost:8000**

端口：宿主机 **8000** → 容器 8000（在 `docker-compose.yml` 的 `ports` 中可改）。
上传的图片与会话保存在宿主机 `./data` 目录（挂载卷），容器重启后仍然存在。

停止：

```bash
docker compose down
```

## 本地直接运行（不用 Docker）

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

访问 http://localhost:8000 。

## 使用步骤

1. **上传照片**：点击右上角按钮，或将图片拖入页面（EXIF 方向会被自动纠正）。
2. **透视标定**：
   - 在侧栏填入参照物的**实际宽、实际高与单位**（mm / cm / m / in / px）；
   - 按 **左上 → 右上 → 右下 → 左下** 的顺序在图上单击四角（与矩形在照片中
     的实际朝向无关，顺序一致即可）；
   - 角点可随时**拖动微调**，标定实时重新计算，有效时四边形变为绿色。
3. **绘制区域**：切换到“绘制”模式，逐点单击多边形顶点；
   - **双击 / Enter / 点击第一个顶点**闭合；
   - `Esc` 取消当前草稿，`Backspace` 撤销最后一个顶点；
   - 可绘制任意多个多边形。
4. **查看与编辑**：
   - 右侧实时显示每个区域校正后的**面积、周长**，可切换结果显示单位
     （mm² / cm² / m² / in² / px²，按长度单位平方自动换算）；
   - “编辑”模式可拖动顶点、拖动整个区域，**右键顶点删除**；
   - 每个区域可重命名、单独删除；“清空区域”一键全部重画。
5. **导出**：点击“导出 JSON”，下载本次全部测量结果（含单应矩阵、图像坐标、
   校正后坐标、面积与周长及校验状态）。

### 鼠标与快捷键

| 操作 | 方式 |
| --- | --- |
| 缩放 | 滚轮（以光标为中心）、`+` / `−`、左下缩放条、`F` 适合窗口、`1` 100% |
| 平移 | “平移”模式拖动、任意模式按住**空格**拖动、或**右键拖动** |
| 模式 | `1` 角点 · `2` 绘制 · `3` 编辑 · `4` 平移 |
| 绘制 | 单击加点，双击/Enter 闭合，Backspace 撤销，Esc 取消 |
| 编辑 | 左键拖动顶点/区域，右键删除顶点，`Delete` 删除选中区域 |

## 什么时候会阻止计算（以及原因）

校准与测量在**前端实时拦截**并由**后端再次权威校验**，错误信息直接显示在界面上：

- 角点重合、三点共线、四点不构成凸四边形（顺序错乱/自交）→
  *“四个角点没有构成凸四边形（顺序错乱或自交），请按 左上→右上→右下→左下 依次点击”*；
- DLT 方程组秩亏损（四点共线等退化情况）→
  *“透视变换不可逆：标定四点共线或构成退化四边形，无法从平面照片恢复唯一映射”*；
- 单应矩阵行列式接近 0，或某点映射到无穷远线（齐次分母 ≈ 0）→
  *“透视矩阵行列式接近 0，变换不可逆……”* / *“……映射时分母接近 0（落在无穷远线上）”*；
- 标定对应残差过大（角点与已知矩形对不上）→
  *“透视校准残差过大……请检查点击顺序”*；
- 多边形顶点重合或**边自交** →
  *“多边形自交（边 i 与边 j 相交），自交多边形没有唯一的面积，请调整顶点”*；
- 多边形顶点经透视变换后落在无穷远线附近 →
  *“多边形顶点 k 的透视分母接近 0……无法计算校正后坐标”*。

数值阈值集中在 `backend/app/geometry.py` 与 `frontend/app.js` 顶部的常量中
（重合、共线、主元、残差、行列式、分母容差），两处保持一致。

## 导出 JSON 结构

```json
{
  "schema": "photo-ruler/1.0",
  "exported_at": "2026-09-19T08:11:43Z",
  "session_id": "…",
  "image": { "filename": "….jpg", "width": 800, "height": 600 },
  "calibration": {
    "corners": [{ "u": 0, "v": 0 } /* …4 个图像像素角点 */],
    "rect_width": 85.6, "rect_height": 53.98, "unit": "mm",
    "homography_image_to_world": [[/* 3×3 */]],
    "valid": true, "errors": []
  },
  "all_measurements_valid": true,
  "polygons": [
    {
      "name": "测试矩形",
      "image_points": [{ "u": 150, "v": 150 } /* … */],
      "world_points": [{ "x": 7.13, "y": 6.75 } /* …校正后坐标 */],
      "area": 433.1895, "perimeter": 83.285,
      "valid": true, "errors": []
    }
  ]
}
```

面积/周长单位为标定所填单位（其平方）。`all_measurements_valid=false` 表示
标定或至少一个多边形无效，对应条目的 `errors` 给出原因。

## 算法说明

- 设图像点 `p=(u,v,1)`、实面坐标 `X=(x,y,1)`，映射 `X ∼ H p`。
- 4 对点给出 8 个线性方程（DLT），对两组点做各向同性归一化后以 SVD
  （后端 NumPy）/ 完全选主元 Gauss-Jordan（前端，无第三方库）求零向量，
  再反归一化得到 `H`。
- 校验链：点重合 → 凸性（叉积同号）→ 主元/秩 → DLT 残差 →
  `|det H|` → 逐点齐次分母。
- 多边形：各顶点经 `H` 变换到实面后做自交检测（非相邻边跨立试验），
  再以鞋带公式求面积（取绝对值）、边长累加求周长。

## API 摘要

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/api/sessions` | 创建会话 |
| GET | `/api/sessions/{id}` | 获取会话（标定 + 各多边形测量值） |
| POST | `/api/sessions/{id}/image` | 上传/替换照片（multipart，清空旧几何） |
| PUT | `/api/sessions/{id}/calibration` | 提交四角与实际宽高，返回 H 与全部区域复算结果 |
| POST/PUT/DELETE | `/api/sessions/{id}/polygons[/{pid}]` | 新增 / 更新 / 删除区域；DELETE 集合路径清空 |
| GET | `/api/sessions/{id}/export` | 下载测量 JSON（attachment） |
| GET | `/api/health` | 健康检查（容器 HEALTHCHECK） |

## 项目结构

```
.
├── Dockerfile
├── docker-compose.yml
├── backend/
│   ├── requirements.txt
│   └── app/
│       ├── main.py          # FastAPI：会话/上传/标定/多边形/导出
│       └── geometry.py      # DLT 单应矩阵、可逆性检查、自交与面积周长
├── frontend/
│   ├── index.html
│   ├── styles.css
│   └── app.js               # 画布、视图变换、交互、镜像几何算法
├── data/                    # 运行时生成：images/ 与 sessions/（挂载卷）
└── tests/
    ├── test_geometry.py     # 合成透视相机数值校验（NumPy）
    ├── test_api.py          # FastAPI 端到端：上传→标定→多边形→导出
    └── parity_geometry.mjs  # 前端 JS 几何与后端数值一致性（Node）
```

## 测试

```bash
pip install -r backend/requirements.txt httpx
python3 tests/test_geometry.py
python3 tests/test_api.py
node tests/parity_geometry.mjs
```

> 说明：本应用用于平面场景（待测物与参照矩形基本共面）。若待测面与参照矩形
> 不在同一平面，单应变换不成立，需要另一套多视图/三维标定方案。
