# HIPAA RAG Demo
这是一个基于 LangChain 和 Chroma 构建的医疗 RAG（Retrieval-Augmented Generation）系统，专为处理 HL7 FHIR 或 HIPAA JSON 数据设计。

## 功能

- 支持解析 FHIR Bundle 或标准 JSON 文件
- 自动提取资源类型、患者ID、Bundle类型等元数据
- 支持向量化检索 + CrossEncoder reranking
- 支持按患者/资源类型过滤查询
- 提供患者时间线排序功能
- 内置 Gemini embedding + LLM 问答链

## 📁 项目结构

- `hipaa_rag.py`：主程序，包含全部类和主函数
- `requirements.txt`：依赖包
- `data/`：用于测试的 FHIR JSON 数据，均为 [Synthea](https://github.com/synthetichealth/synthea) 生成的合成数据，不含真实患者信息

## 🚀 快速使用

```bash
# 安装依赖
pip install -r requirements.txt

# 运行 demo
python hipaa_rag.py
