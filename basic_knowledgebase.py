import os
import json
from typing import List, Dict, Any
from pathlib import Path

from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_deepseek import ChatDeepSeek
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import (
    TextFileLoader, 
    PyPDFLoader, 
    CSVLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredMarkdownLoader
)
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

# 加载环境变量
load_dotenv()

class DocumentKnowledgeBase:
    def __init__(self):
        """初始化知识库"""
        self.llm = ChatDeepSeek(
            api_key=os.getenv("DEEPSEEK_API_KEY"), 
            model="deepseek-chat",
            temperature=0.1
        )
        
        # 使用免费的HuggingFace嵌入模型
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        
        self.vectorstore = None
        self.documents = []
        
        # 创建存储目录
        self.storage_dir = Path("knowledge_base")
        self.storage_dir.mkdir(exist_ok=True)
        
        # 加载已有的知识库
        self.load_existing_knowledge_base()
    
    def load_document(self, file_path: str) -> List[Document]:
        """根据文件类型加载文档"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        file_extension = file_path.suffix.lower()
        
        try:
            if file_extension == '.txt':
                loader = TextFileLoader(str(file_path), encoding='utf-8')
            elif file_extension == '.pdf':
                loader = PyPDFLoader(str(file_path))
            elif file_extension == '.csv':
                loader = CSVLoader(str(file_path))
            elif file_extension in ['.doc', '.docx']:
                loader = UnstructuredWordDocumentLoader(str(file_path))
            elif file_extension == '.md':
                loader = UnstructuredMarkdownLoader(str(file_path))
            else:
                # 尝试作为文本文件处理
                loader = TextFileLoader(str(file_path), encoding='utf-8')
            
            documents = loader.load()
            
            # 为文档添加来源信息
            for doc in documents:
                doc.metadata['source'] = str(file_path)
                doc.metadata['filename'] = file_path.name
            
            return documents
            
        except Exception as e:
            print(f"❌ 加载文件失败 {file_path}: {e}")
            return []
    
    def add_documents(self, file_paths: List[str]):
        """添加多个文档到知识库"""
        new_documents = []
        
        for file_path in file_paths:
            print(f"📄 正在加载: {file_path}")
            docs = self.load_document(file_path)
            if docs:
                new_documents.extend(docs)
                print(f"✅ 成功加载: {file_path} ({len(docs)} 个文档片段)")
            else:
                print(f"❌ 加载失败: {file_path}")
        
        if new_documents:
            # 分割文档
            print("📝 正在分割文档...")
            split_docs = self.text_splitter.split_documents(new_documents)
            
            # 添加到现有文档列表
            self.documents.extend(split_docs)
            
            # 更新或创建向量存储
            if self.vectorstore is None:
                print("🔍 正在创建向量索引...")
                self.vectorstore = FAISS.from_documents(split_docs, self.embeddings)
            else:
                print("🔍 正在更新向量索引...")
                self.vectorstore.add_documents(split_docs)
            
            # 保存知识库
            self.save_knowledge_base()
            
            print(f"✅ 成功添加 {len(split_docs)} 个文档片段到知识库")
            print(f"📊 知识库总文档数: {len(self.documents)}")
        else:
            print("❌ 没有成功加载任何文档")
    
    def save_knowledge_base(self):
        """保存知识库到本地"""
        try:
            if self.vectorstore:
                # 保存FAISS索引
                faiss_path = self.storage_dir / "faiss_index"
                self.vectorstore.save_local(str(faiss_path))
                
                # 保存文档元数据
                doc_metadata = []
                for doc in self.documents:
                    doc_metadata.append({
                        'content': doc.page_content,
                        'metadata': doc.metadata
                    })
                
                metadata_path = self.storage_dir / "documents_metadata.json"
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(doc_metadata, f, ensure_ascii=False, indent=2)
                
                print("💾 知识库已保存到本地")
        except Exception as e:
            print(f"❌ 保存知识库失败: {e}")
    
    def load_existing_knowledge_base(self):
        """加载已有的知识库"""
        try:
            faiss_path = self.storage_dir / "faiss_index"
            metadata_path = self.storage_dir / "documents_metadata.json"
            
            if faiss_path.exists() and metadata_path.exists():
                # 加载FAISS索引
                self.vectorstore = FAISS.load_local(
                    str(faiss_path), 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                
                # 加载文档元数据
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    doc_metadata = json.load(f)
                
                self.documents = [
                    Document(page_content=item['content'], metadata=item['metadata'])
                    for item in doc_metadata
                ]
                
                print(f"📚 已加载现有知识库，包含 {len(self.documents)} 个文档片段")
        except Exception as e:
            print(f"⚠️ 加载现有知识库失败: {e}")
    
    def search_documents(self, query: str, k: int = 5) -> List[Document]:
        """搜索相关文档"""
        if not self.vectorstore:
            return []
        
        try:
            docs = self.vectorstore.similarity_search(query, k=k)
            return docs
        except Exception as e:
            print(f"❌ 搜索文档失败: {e}")
            return []
    
    def query(self, question: str) -> str:
        """基于知识库回答问题"""
        if not self.vectorstore:
            return "❌ 知识库为空，请先添加文档。"
        
        # 搜索相关文档
        relevant_docs = self.search_documents(question, k=3)
        
        if not relevant_docs:
            return "❌ 未找到相关文档内容。"
        
        # 构建上下文
        context = "\n\n".join([
            f"文档片段 {i+1} (来源: {doc.metadata.get('filename', '未知')}):\n{doc.page_content}"
            for i, doc in enumerate(relevant_docs)
        ])
        
        # 创建提示模板
        prompt_template = PromptTemplate.from_template("""
基于以下文档内容回答问题。请确保答案准确且基于提供的文档内容。

文档内容:
{context}

问题: {question}

请基于上述文档内容回答问题。如果文档中没有相关信息，请明确说明。
回答:""")
        
        # 创建链并执行
        chain = LLMChain(llm=self.llm, prompt=prompt_template)
        
        try:
            response = chain.run({
                "context": context,
                "question": question
            })
            
            # 添加来源信息
            sources = list(set([doc.metadata.get('filename', '未知') for doc in relevant_docs]))
            response += f"\n\n📚 参考来源: {', '.join(sources)}"
            
            return response
        except Exception as e:
            return f"❌ 生成回答时出错: {e}"
    
    def list_documents(self):
        """列出知识库中的所有文档"""
        if not self.documents:
            print("📚 知识库为空")
            return
        
        # 统计文档来源
        sources = {}
        for doc in self.documents:
            filename = doc.metadata.get('filename', '未知')
            sources[filename] = sources.get(filename, 0) + 1
        
        print("📚 知识库文档列表:")
        for filename, count in sources.items():
            print(f"  - {filename}: {count} 个片段")
        
        print(f"\n📊 总计: {len(sources)} 个文件, {len(self.documents)} 个文档片段")
    
    def clear_knowledge_base(self):
        """清空知识库"""
        self.documents = []
        self.vectorstore = None
        
        # 删除本地存储的文件
        try:
            import shutil
            if self.storage_dir.exists():
                shutil.rmtree(self.storage_dir)
                self.storage_dir.mkdir(exist_ok=True)
            print("🗑️ 知识库已清空")
        except Exception as e:
            print(f"❌ 清空知识库失败: {e}")

def main():
    """主程序"""
    print("📚 基于文档的知识库工具")
    print("=" * 50)
    
    kb = DocumentKnowledgeBase()
    
    while True:
        print("\n可用命令:")
        print("1. add - 添加文档到知识库")
        print("2. query - 基于知识库问答")
        print("3. list - 查看知识库文档")
        print("4. clear - 清空知识库")
        print("5. exit - 退出程序")
        
        command = input("\n请选择命令 (1-5): ").strip()
        
        if command == "1" or command.lower() == "add":
            file_paths = input("请输入文档路径 (多个文件用逗号分隔): ").strip()
            if file_paths:
                paths = [path.strip() for path in file_paths.split(",")]
                kb.add_documents(paths)
        
        elif command == "2" or command.lower() == "query":
            question = input("请输入您的问题: ").strip()
            if question:
                print("\n🤔 正在思考...")
                response = kb.query(question)
                print(f"\n💡 回答:\n{response}")
        
        elif command == "3" or command.lower() == "list":
            kb.list_documents()
        
        elif command == "4" or command.lower() == "clear":
            confirm = input("确定要清空知识库吗? (y/n): ").strip().lower()
            if confirm == 'y':
                kb.clear_knowledge_base()
        
        elif command == "5" or command.lower() == "exit":
            print("👋 再见!")
            break
        
        else:
            print("❌ 无效命令，请重新选择")

if __name__ == "__main__":
    main()
