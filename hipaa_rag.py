#!/usr/bin/env python3
"""
HIPAA医疗文档智能问答系统
使用LangChain和Chroma向量数据库处理FHIR/HIPAA JSON文档
"""

import os
import json
import torch
from pathlib import Path

import numpy as np
from langchain.text_splitter import RecursiveJsonSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from sentence_transformers import CrossEncoder


class HIPAADocumentRAG:
    """HIPAA医疗文档智能问答系统"""
    
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        """
        初始化HIPAA文档处理系统
        
        参数:
            chunk_size: 文档块的最大大小
            chunk_overlap: 文档块之间的重叠大小
        """
        # 设置基本参数
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # 是否使用重新排序功能
        self.use_reranker = True
        
        # 初始化谷歌嵌入模型
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-exp-03-07"
        )
        
        # 初始化重新排序模型
        self.reranker = None
        self.setup_reranker()
        
        # 初始化JSON文档分割器
        self.json_splitter = RecursiveJsonSplitter(
            max_chunk_size=chunk_size,
            min_chunk_size=100
        )
        
        # 初始化向量数据库
        self.vector_store = None
        self.documents = []
        
    def setup_reranker(self):
        """设置重新排序模型"""
        try:
            self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            print("重新排序模型初始化成功")
        except Exception as e:
            print(f"警告：重新排序模型初始化失败：{e}")
            self.use_reranker = False
            self.reranker = None
        
    def load_hipaa_documents(self, file_path):
        """
        加载HIPAA/FHIR JSON文档
        
        参数:
            file_path: JSON文件路径或包含JSON文件的目录路径
            
        返回:
            处理后的文档列表
        """
        documents = []
        file_path = Path(file_path)
        
        # 如果是单个文件
        if file_path.is_file() and file_path.suffix == '.json':
            print(f"正在处理单个文件: {file_path}")
            documents.extend(self.process_single_file(file_path))
        # 如果是目录
        elif file_path.is_dir():
            print(f"正在处理目录: {file_path}")
            for json_file in file_path.glob('*.json'):
                documents.extend(self.process_single_file(json_file))
        else:
            print(f"错误：无效的文件路径: {file_path}")
            return []
            
        self.documents = documents
        print(f"成功加载了 {len(documents)} 个文档块")
        return documents
    
    def process_single_file(self, file_path):
        """处理单个JSON文件"""
        try:
            print(f"正在处理文件: {file_path.name}")
            
            # 读取JSON文件
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            # 检查是否是FHIR Bundle格式
            if json_data.get('resourceType') == 'Bundle' and 'entry' in json_data:
                print("检测到FHIR Bundle格式")
                return self.process_fhir_bundle(json_data, file_path)
            else:
                print("使用标准JSON处理")
                return self.process_standard_json(json_data, file_path)
                
        except Exception as e:
            print(f"处理文件时出错 {file_path}: {e}")
            return []
    
    def process_fhir_bundle(self, bundle_data, file_path):
        """处理FHIR Bundle格式的文件"""
        documents = []
        entries = bundle_data.get('entry', [])
        
        print(f"Bundle包含 {len(entries)} 个条目")
        
        # 处理每个条目
        for i, entry in enumerate(entries):
            # 获取资源数据
            resource = entry.get('resource', entry)
            
            # 将资源转换为JSON字符串
            chunk_text = json.dumps(resource, indent=2, ensure_ascii=False)
            
            # 提取元数据
            metadata = {}
            metadata['source'] = str(file_path)
            metadata['chunk_id'] = i
            metadata['entry_index'] = i
            metadata['resource_type'] = resource.get('resourceType')
            metadata['resource_id'] = resource.get('id')
            metadata['patient_id'] = self.extract_patient_id_from_resource(resource)
            metadata['bundle_type'] = bundle_data.get('type')
            metadata['is_bundle_entry'] = True
            
            # 创建文档对象
            doc = Document(
                page_content=chunk_text,
                metadata=metadata
            )
            documents.append(doc)
        
        return documents
    
    def process_standard_json(self, json_data, file_path):
        """处理标准JSON文件"""
        # 使用JSON分割器将数据分块
        json_chunks = self.json_splitter.split_json(json_data)
        
        documents = []
        for i, chunk in enumerate(json_chunks):
            # 将块转换为JSON字符串
            chunk_text = json.dumps(chunk, indent=2, ensure_ascii=False)
            
            # 提取元数据
            metadata = {}
            metadata['source'] = str(file_path)
            metadata['chunk_id'] = i
            metadata['resource_type'] = self.extract_resource_type(chunk)
            metadata['patient_id'] = self.extract_patient_id(chunk)
            metadata['is_bundle_entry'] = False
            
            # 创建文档对象
            doc = Document(
                page_content=chunk_text,
                metadata=metadata
            )
            documents.append(doc)
            
        return documents
    
    def extract_resource_type(self, chunk):
        """从数据块中提取FHIR资源类型"""
        if isinstance(chunk, dict):
            # 先检查直接的resourceType
            if 'resourceType' in chunk:
                return chunk['resourceType']
            # 再检查resource子项中的resourceType
            if 'resource' in chunk and isinstance(chunk['resource'], dict):
                return chunk['resource'].get('resourceType')
        return None
    
    def extract_patient_id(self, chunk):
        """从数据块中提取患者ID"""
        if isinstance(chunk, dict):
            # 尝试不同的可能位置
            patient_id = chunk.get('id')
            
            # 如果没找到，检查resource子项
            if not patient_id and 'resource' in chunk:
                patient_id = chunk['resource'].get('id')
            
            # 检查subject引用
            if not patient_id and 'subject' in chunk and chunk['subject'] is not None:
                subject_ref = chunk['subject'].get('reference', '')
                if subject_ref:
                    patient_id = subject_ref.split('/')[-1]
            
            return patient_id
        return None
    
    def extract_patient_id_from_resource(self, resource):
        """从FHIR资源中提取患者ID"""
        if not isinstance(resource, dict):
            return None
        
        # 如果资源本身就是Patient类型
        if resource.get('resourceType') == 'Patient':
            return resource.get('id')
        
        # 查找subject引用
        subject = resource.get('subject', {})
        if isinstance(subject, dict) and 'reference' in subject:
            ref = subject['reference']
            return self.extract_id_from_reference(ref)
        
        # 查找patient引用
        patient = resource.get('patient', {})
        if isinstance(patient, dict) and 'reference' in patient:
            ref = patient['reference']
            return self.extract_id_from_reference(ref)
        
        # 查找performer引用
        performer = resource.get('performer', [])
        if isinstance(performer, list):
            for perf in performer:
                if isinstance(perf, dict) and 'reference' in perf:
                    ref = perf['reference']
                    if 'Patient/' in ref or 'urn:uuid:' in ref:
                        return self.extract_id_from_reference(ref)
        
        return None
    
    def extract_id_from_reference(self, reference):
        """从FHIR引用字符串中提取ID"""
        if not reference:
            return None
        
        # 处理Patient/id格式
        if reference.startswith('Patient/'):
            return reference.split('/')[-1]
        
        # 处理urn:uuid:id格式
        if reference.startswith('urn:uuid:'):
            return reference.split(':')[-1]
        
        # 处理#id格式
        if reference.startswith('#'):
            return reference[1:]
        
        # 其他格式，尝试提取最后的部分
        if '/' in reference:
            return reference.split('/')[-1]
        
        return reference
    
    def create_vector_store(self, persist_directory="./chroma_db"):
        """
        创建Chroma向量数据库
        
        参数:
            persist_directory: 数据库持久化目录
        
        返回:
            Chroma向量数据库实例
        """
        if not self.documents:
            print("错误：没有加载任何文档。请先调用load_hipaa_documents()")
            return None
        
        print("正在创建向量嵌入...")
        self.vector_store = Chroma.from_documents(
            documents=self.documents,
            embedding=self.embeddings,
            persist_directory=persist_directory
        )
        
        print(f"成功创建向量数据库，包含 {len(self.documents)} 个文档")
        print(f"数据库保存到: {persist_directory}")
        return self.vector_store
    
    def save_vector_store(self):
        """保存向量数据库（Chroma会自动持久化）"""
        if self.vector_store is None:
            print("错误：还没有创建向量数据库")
            return
        
        # Chroma会自动持久化
        self.vector_store.persist()
        print("向量数据库已保存")
    
    def load_vector_store(self, persist_directory="./chroma_db"):
        """
        从磁盘加载向量数据库
        
        参数:
            persist_directory: 数据库持久化目录
        """
        self.vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings
        )
        print(f"向量数据库已从 {persist_directory} 加载")
    
    def rerank_documents(self, query, documents, top_k=None):
        """
        使用重新排序模型提高搜索结果的相关性
        
        参数:
            query: 查询字符串
            documents: 待排序的文档列表
            top_k: 返回的文档数量
            
        返回:
            重新排序后的文档列表
        """
        # 如果没有启用重新排序或没有文档
        if not self.use_reranker or self.reranker is None or not documents:
            return documents
        
        if top_k is None:
            top_k = len(documents)
        
        # 准备查询-文档对
        pairs = []
        for doc in documents:
            # 为了提高效率，只使用前500个字符
            content = doc.page_content[:500]
            pairs.append([query, content])
        
        # 获取重新排序分数
        try:
            scores = self.reranker.predict(pairs)
            
            # 将文档和分数组合并排序
            doc_scores = []
            for i, doc in enumerate(documents):
                doc_scores.append((doc, scores[i]))
            
            # 按分数降序排序
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            
            # 返回前top_k个文档
            reranked_docs = []
            for i in range(min(top_k, len(doc_scores))):
                reranked_docs.append(doc_scores[i][0])
            
            print(f"重新排序了 {len(documents)} 个文档，返回前 {len(reranked_docs)} 个")
            return reranked_docs
            
        except Exception as e:
            print(f"警告：重新排序失败：{e}")
            return documents[:top_k]
    
    def query(self, question, k=5, use_rerank=True, filter_metadata=None):
        """
        查询向量数据库获取相关文档
        
        参数:
            question: 查询问题
            k: 返回的文档数量
            use_rerank: 是否使用重新排序
            filter_metadata: 元数据过滤条件
            
        返回:
            相关文档列表
        """
        if self.vector_store is None:
            print("错误：还没有创建或加载向量数据库")
            return []
        
        # 如果要使用重新排序，先获取更多文档
        if use_rerank and self.use_reranker:
            initial_k = k * 3
        else:
            initial_k = k
        
        # 初始检索
        try:
            if filter_metadata:
                # Chroma使用'where'参数进行元数据过滤
                initial_docs = self.vector_store.similarity_search(
                    question, 
                    k=initial_k, 
                    where=filter_metadata
                )
            else:
                initial_docs = self.vector_store.similarity_search(question, k=initial_k)
            
            # 如果启用重新排序且有多个文档
            if use_rerank and self.use_reranker and len(initial_docs) > 1:
                reranked_docs = self.rerank_documents(question, initial_docs, k)
                return reranked_docs
            else:
                return initial_docs[:k]
                
        except Exception as e:
            print(f"查询时出错：{e}")
            return []
    
    def create_qa_chain(self, llm_model="gemini-2.5-flash-preview-04-17"):
        """
        创建问答链
        
        参数:
            llm_model: 使用的语言模型
            
        返回:
            RetrievalQA问答链
        """
        if self.vector_store is None:
            print("错误：还没有创建向量数据库")
            return None
        
        # 初始化语言模型
        llm = ChatGoogleGenerativeAI(model=llm_model, temperature=0)
        
        # 创建问答链
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 5}),
            return_source_documents=True
        )
        
        return qa_chain
    
    def search_by_resource_type(self, resource_type, k=5):
        """按FHIR资源类型搜索文档"""
        filter_condition = {"resource_type": resource_type}
        return self.query(
            question=f"resourceType: {resource_type}",
            k=k,
            filter_metadata=filter_condition
        )
    
    def search_by_patient(self, patient_id, k=10):
        """搜索特定患者的所有相关文档"""
        filter_condition = {"patient_id": patient_id}
        return self.query(
            question=f"patient ID: {patient_id}",
            k=k,
            filter_metadata=filter_condition
        )
    
    def get_bundle_statistics(self):
        """
        获取已加载文档的统计信息
        
        返回:
            包含统计信息的字典
        """
        if not self.documents:
            return {"error": "没有加载任何文档"}
        
        # 初始化统计数据
        stats = {
            "total_documents": len(self.documents),
            "bundle_entries": 0,
            "non_bundle_entries": 0,
            "resource_types": {},
            "patients": [],
            "bundle_types": []
        }
        
        # 统计各种信息
        patient_set = set()
        bundle_type_set = set()
        
        for doc in self.documents:
            metadata = doc.metadata
            
            # 统计Bundle vs 非Bundle条目
            if metadata.get('is_bundle_entry', False):
                stats["bundle_entries"] += 1
            else:
                stats["non_bundle_entries"] += 1
            
            # 统计资源类型
            resource_type = metadata.get('resource_type', 'Unknown')
            if resource_type in stats["resource_types"]:
                stats["resource_types"][resource_type] += 1
            else:
                stats["resource_types"][resource_type] = 1
            
            # 收集唯一患者ID
            patient_id = metadata.get('patient_id')
            if patient_id:
                patient_set.add(patient_id)
            
            # 收集Bundle类型
            bundle_type = metadata.get('bundle_type')
            if bundle_type:
                bundle_type_set.add(bundle_type)
        
        # 转换为列表
        stats["patients"] = list(patient_set)
        stats["bundle_types"] = list(bundle_type_set)
        stats["unique_patients"] = len(stats["patients"])
        
        return stats
    
    def search_by_resource_id(self, resource_id, k=5):
        """按特定的FHIR资源ID搜索文档"""
        filter_condition = {"resource_id": resource_id}
        return self.query(
            question=f"resource ID: {resource_id}",
            k=k,
            filter_metadata=filter_condition
        )
    
    def get_patient_timeline(self, patient_id, sort_by_date=True):
        """
        获取特定患者的所有文档，可按日期排序
        
        参数:
            patient_id: 患者ID
            sort_by_date: 是否按日期排序
            
        返回:
            患者相关的文档列表
        """
        # 获取患者的所有文档
        patient_docs = self.search_by_patient(patient_id, k=100)
        
        # 如果需要按日期排序
        if sort_by_date and patient_docs:
            def get_date_for_sorting(doc):
                """提取文档中的日期用于排序"""
                try:
                    content = json.loads(doc.page_content)
                    
                    # 常见的日期字段
                    date_fields = [
                        'effectiveDateTime', 'issued', 'date', 'performedDateTime', 
                        'authoredOn', 'onsetDateTime'
                    ]
                    
                    # 查找日期字段
                    for field in date_fields:
                        if field in content:
                            date_val = content[field]
                            if date_val:
                                return date_val
                    
                    # 处理嵌套的日期字段
                    if 'period' in content and isinstance(content['period'], dict):
                        start_date = content['period'].get('start')
                        if start_date:
                            return start_date
                    
                    return "1900-01-01"  # 默认日期
                except:
                    return "1900-01-01"
            
            # 按日期排序
            patient_docs.sort(key=get_date_for_sorting)
        
        return patient_docs


def main():
    """使用示例"""
    
    print("=== HIPAA文档智能问答系统 ===")
    
    # 创建RAG系统
    rag = HIPAADocumentRAG(chunk_size=800, chunk_overlap=100)
    
    # 加载HIPAA文档
    documents_path = "./file"  # 包含JSON文件的目录
    documents = rag.load_hipaa_documents(documents_path)
    
    if not documents:
        print("没有找到任何文档")
        return
    
    # 显示统计信息
    print("\n=== 文档处理统计 ===")
    stats = rag.get_bundle_statistics()
    print(f"总文档数: {stats['total_documents']}")
    print(f"Bundle条目: {stats['bundle_entries']}")
    print(f"非Bundle条目: {stats['non_bundle_entries']}")
    print(f"唯一患者数: {stats['unique_patients']}")
    print(f"发现的资源类型数: {len(stats['resource_types'])}")
    
    print("\n资源类型分布:")
    for resource_type, count in sorted(stats['resource_types'].items()):
        print(f"  {resource_type}: {count}")
    
    # 创建向量数据库
    vector_store = rag.create_vector_store()
    
    if vector_store is None:
        print("创建向量数据库失败")
        return
    
    # 示例查询
    print("\n=== 查询示例 ===")
    
    # 基本相似度搜索
    results = rag.query("patient information and demographics", k=3)
    print(f"\n找到 {len(results)} 个与患者人口统计相关的文档:")
    for i, doc in enumerate(results):
        print(f"  {i+1}. 资源类型: {doc.metadata.get('resource_type', 'Unknown')}")
        print(f"     患者ID: {doc.metadata.get('patient_id', 'Unknown')}")
        print(f"     资源ID: {doc.metadata.get('resource_id', 'Unknown')}")
        print(f"     是Bundle条目: {doc.metadata.get('is_bundle_entry', False)}")
    
    # 按资源类型搜索
    patient_docs = rag.search_by_resource_type("Patient", k=2)
    print(f"\n找到 {len(patient_docs)} 个Patient资源文档")
    
    # 如果有患者，显示第一个患者的时间线
    if stats['patients']:
        first_patient = stats['patients'][0]
        print(f"\n=== 患者 {first_patient} 的时间线 ===")
        timeline = rag.get_patient_timeline(first_patient)
        print(f"找到 {len(timeline)} 个与患者 {first_patient} 相关的文档")
        for i, doc in enumerate(timeline[:5]):  # 显示前5个
            resource_type = doc.metadata.get('resource_type', 'Unknown')
            print(f"  {i+1}. {resource_type}")
    
    # 按特定资源类型搜索
    print(f"\n=== 按资源类型搜索 ===")
    for resource_type in ['Observation', 'Encounter', 'MedicationRequest']:
        if resource_type in stats['resource_types']:
            docs = rag.search_by_resource_type(resource_type, k=2)
            print(f"找到 {len(docs)} 个 {resource_type} 文档")
    
    print("\nRAG系统设置完成！")


if __name__ == "__main__":
    main()