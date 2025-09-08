#!/usr/bin/env python3
"""
基于LangExtract的MIMIC论文信息提取器 - 核心实现
从医学论文中提取结构化的复现任务信息

作者：MedResearcher项目
创建时间：2025-01-25
"""

import langextract as lx
import textwrap
from pathlib import Path
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# 配置日志
logger = logging.getLogger(__name__)


class MIMICLangExtractBuilder:
    """基于LangExtract的MIMIC论文信息提取器"""
    
    def __init__(self, doc_workers: int = 4):
        """初始化提取器，配置vllm API服务
        
        Args:
            doc_workers: 文档并行处理工作线程数，默认为4
        """
        try:
            # 配置LangExtract使用vllm API（通过OpenAI兼容接口）
            import os
            os.environ["LANGEXTRACT_API_KEY"] = "dummy"
            
            # 创建ModelConfig，强制使用OpenAI提供者访问vllm端点
            # self.model_config = lx.factory.ModelConfig(
            #     model_id="gpt-oss",  # 使用vllm中实际部署的模型名称
            #     provider="OpenAILanguageModel",  # 强制指定OpenAI提供者
            #     provider_kwargs={
            #         "base_url": "http://192.168.31.127:19090/v1",  # vllm API端点
            #         "api_key": "dummy",
            #         "model_id": "gpt-oss-20b"  # 确保使用正确的模型ID
            #     }
            # )
            self.model_config = lx.factory.ModelConfig(
                model_id="gpt-oss",               # 与 vLLM 部署的模型名称一致
                provider="OpenAI",                 # 强制使用 OpenAI 兼容 provider
                provider_kwargs={
                    "base_url": "http://192.168.31.127:19090/v1",  # vLLM API 地址
                    "api_key": "gpustack_d402860477878812_9ec494a501497d25b565987754f4db8c",
                    "model_id": "gpt-oss"         # 与 model_id 保持一致
                }
            )

            
            # LangExtract通用配置参数
            self.extract_config = {
                "config": self.model_config,
                "max_workers": 5,          # 降低并发，避免过载vllm服务
                "max_char_buffer": 6000,   # 适合医学论文的上下文长度
                "extraction_passes": 1,    # 单次提取，避免过多API调用
                "temperature": 0.1,        # 较低温度确保一致性
                "fence_output": True,      # 期望代码围栏格式输出
                "use_schema_constraints": False,  # vllm可能不支持严格schema
                "debug": False
            }
            
            # 加载所有模块的提取配置
            self.module_configs = {
                "data": self._load_data_config(),
                "model": self._load_model_config(), 
                "training": self._load_training_config(),
                "evaluation": self._load_evaluation_config(),
                "environment": self._load_environment_config()
            }
            
            # 文档并行处理配置
            self.doc_workers = max(1, doc_workers)  # 确保至少有1个工作线程
            self.progress_lock = threading.Lock()   # 保护进度保存操作的线程锁
            
            logger.info(f"MIMICLangExtractBuilder初始化成功 (文档并行度: {self.doc_workers})")
            
        except Exception as e:
            logger.error(f"初始化失败: {e}")
            raise
    
    def _load_data_config(self) -> Dict[str, Any]:
        """加载数据模块的LangExtract配置"""
        return {
            "prompt": textwrap.dedent("""
                Extract specific data processing information from medical papers. Follow these rules strictly:
                
                1. dataset_source: Extract clearly mentioned dataset names (e.g., "MIMIC-IV", "Stanford EHR")
                2. data_scale: Extract specific data scale numbers (e.g., "135,483 patients", "2015-2023")  
                3. preprocessing_step: Extract specific descriptions of data preprocessing steps
                4. feature_type: Extract descriptions of feature types and encoding methods
                5. inclusion_criteria: Extract exact text of patient inclusion criteria
                6. exclusion_criteria: Extract exact text of patient exclusion criteria
                
                Use exact text for extraction, do not paraphrase. Provide meaningful attributes for each extraction.
                """),
            "examples": [
                lx.data.ExampleData(
                    text="We analyzed 135,483 ED blood culture orders from Stanford Medicine EHR between 2015-2023. Adult patients (≥18 years) with blood culture collection in the ED were included. Patients with positive blood cultures within 14 days were excluded. Features were one-hot encoded for ML compatibility.",
                    extractions=[
                        lx.data.Extraction(
                            extraction_class="dataset_source",
                            extraction_text="Stanford Medicine EHR",
                            attributes={
                                "data_type": "electronic health records", 
                                "institution": "Stanford Medicine"
                            }
                        ),
                        lx.data.Extraction(
                            extraction_class="data_scale", 
                            extraction_text="135,483 ED blood culture orders",
                            attributes={
                                "sample_size": "135,483", 
                                "time_range": "2015-2023", 
                                "data_unit": "blood culture orders"
                            }
                        ),
                        lx.data.Extraction(
                            extraction_class="inclusion_criteria",
                            extraction_text="Adult patients (≥18 years) with blood culture collection in the ED",
                            attributes={
                                "age_limit": "≥18 years",
                                "setting": "Emergency Department",
                                "requirement": "blood culture collection"
                            }
                        ),
                        lx.data.Extraction(
                            extraction_class="exclusion_criteria",
                            extraction_text="Patients with positive blood cultures within 14 days were excluded", 
                            attributes={
                                "timeframe": "within 14 days",
                                "condition": "positive blood cultures"
                            }
                        ),
                        lx.data.Extraction(
                            extraction_class="feature_type",
                            extraction_text="Features were one-hot encoded for ML compatibility",
                            attributes={
                                "encoding_method": "one-hot encoding",
                                "purpose": "ML compatibility"
                            }
                        )
                    ]
                ),
                lx.data.ExampleData(
                    text="This study utilized MIMIC-IV database, including CHARTEVENTS and LABEVENTS tables. We extracted hourly vital signs and laboratory values for ICU patients. Missing values were imputed using forward-fill method. Outliers beyond 3 standard deviations were removed.",
                    extractions=[
                        lx.data.Extraction(
                            extraction_class="dataset_source",
                            extraction_text="MIMIC-IV database",
                            attributes={
                                "data_type": "public clinical database",
                                "tables": "CHARTEVENTS, LABEVENTS"
                            }
                        ),
                        lx.data.Extraction(
                            extraction_class="preprocessing_step",
                            extraction_text="Missing values were imputed using forward-fill method",
                            attributes={
                                "method": "forward-fill",
                                "target": "missing values"
                            }
                        ),
                        lx.data.Extraction(
                            extraction_class="preprocessing_step", 
                            extraction_text="Outliers beyond 3 standard deviations were removed",
                            attributes={
                                "method": "outlier removal",
                                "threshold": "3 standard deviations"
                            }
                        )
                    ]
                )
            ]
        }
    
    def _load_model_config(self) -> Dict[str, Any]:
        """加载模型模块的LangExtract配置"""
        return {
            "prompt": textwrap.dedent("""
                Extract specific machine learning model information from medical papers. Follow these rules strictly:
                
                1. model_name: Extract clearly mentioned model names (e.g., "XGBoost", "LSTM", "GPT-4")
                2. architecture_detail: Extract specific text describing architecture
                3. hyperparameter: Extract specific numerical values of hyperparameter settings
                4. feature_processing: Extract descriptions of feature processing methods
                5. model_component: Extract descriptions of model components or modules
                
                Use exact text for extraction, do not paraphrase. Provide meaningful attributes for each extraction.
                """),
            "examples": [
                lx.data.ExampleData(
                    text="We employed XGBoost classifier with max depth of 4 and 30 boosting iterations. Class weights were used to handle imbalanced data. STELLA 1.5B model was used for text embeddings with attention-weighted average pooling.",
                    extractions=[
                        lx.data.Extraction(
                            extraction_class="model_name",
                            extraction_text="XGBoost classifier",
                            attributes={
                                "model_type": "gradient boosting",
                                "task": "classification"
                            }
                        ),
                        lx.data.Extraction(
                            extraction_class="hyperparameter",
                            extraction_text="max depth of 4 and 30 boosting iterations",
                            attributes={
                                "max_depth": "4",
                                "n_estimators": "30",
                                "parameter_type": "tree_structure"
                            }
                        ),
                        lx.data.Extraction(
                            extraction_class="model_name",
                            extraction_text="STELLA 1.5B model",
                            attributes={
                                "model_type": "pretrained language model",
                                "parameters": "1.5B",
                                "purpose": "text embeddings"
                            }
                        ),
                        lx.data.Extraction(
                            extraction_class="feature_processing",
                            extraction_text="attention-weighted average pooling",
                            attributes={
                                "technique": "pooling",
                                "method": "attention-weighted"
                            }
                        )
                    ]
                )
            ]
        }
    
    def _load_training_config(self) -> Dict[str, Any]:
        """加载训练模块的LangExtract配置"""
        return {
            "prompt": textwrap.dedent("""
                Extract specific model training information from medical papers. Follow these rules strictly:
                
                1. data_split_method: Extract specific descriptions of data splitting methods
                2. validation_approach: Extract specific descriptions of validation strategies
                3. hyperparameter_tuning: Extract hyperparameter tuning methods
                4. stopping_condition: Extract training stopping conditions
                5. optimizer_config: Extract optimizer configuration information
                
                Use exact text for extraction, do not paraphrase. Provide meaningful attributes for each extraction.
                """),
            "examples": [
                lx.data.ExampleData(
                    text="Data was split temporally: training set (2015-2022), development set (2022-2023) for hyperparameter tuning, and evaluation set (2023+). Grid search was performed on the development set to optimize AUC performance. Early stopping was applied when validation loss did not improve for 10 epochs.",
                    extractions=[
                        lx.data.Extraction(
                            extraction_class="data_split_method",
                            extraction_text="Data was split temporally: training set (2015-2022), development set (2022-2023), and evaluation set (2023+)",
                            attributes={
                                "split_type": "temporal",
                                "train_period": "2015-2022",
                                "dev_period": "2022-2023",
                                "eval_period": "2023+"
                            }
                        ),
                        lx.data.Extraction(
                            extraction_class="hyperparameter_tuning",
                            extraction_text="Grid search was performed on the development set to optimize AUC performance",
                            attributes={
                                "method": "grid search",
                                "metric": "AUC",
                                "dataset": "development set"
                            }
                        ),
                        lx.data.Extraction(
                            extraction_class="stopping_condition",
                            extraction_text="Early stopping was applied when validation loss did not improve for 10 epochs",
                            attributes={
                                "method": "early stopping",
                                "patience": "10 epochs",
                                "monitor": "validation loss"
                            }
                        )
                    ]
                )
            ]
        }
    
    def _load_evaluation_config(self) -> Dict[str, Any]:
        """加载评估模块的LangExtract配置"""
        return {
            "prompt": textwrap.dedent("""
                Extract specific model evaluation information from medical papers. Follow these rules strictly:
                
                1. evaluation_metric: Extract specific evaluation metric names (e.g., "AUC", "F1-score", "sensitivity")
                2. baseline_comparison: Extract descriptions of baseline models or methods
                3. performance_result: Extract specific numerical performance results
                4. statistical_test: Extract descriptions of statistical testing methods
                5. experimental_setting: Extract specific information about experimental settings
                
                Use exact text for extraction, do not paraphrase. Provide meaningful attributes for each extraction.
                """),
            "examples": [
                lx.data.ExampleData(
                    text="The model achieved ROC-AUC of 0.85 (95% CI: 0.82-0.88) on the test set. We compared against three baselines: expert framework (manual assessment), structured-only model, and LLM-automated framework. At 90% sensitivity, our model achieved 45% specificity versus 32% for the baseline.",
                    extractions=[
                        lx.data.Extraction(
                            extraction_class="evaluation_metric",
                            extraction_text="ROC-AUC",
                            attributes={
                                "metric_type": "discriminative performance",
                                "range": "0-1"
                            }
                        ),
                        lx.data.Extraction(
                            extraction_class="performance_result",
                            extraction_text="ROC-AUC of 0.85 (95% CI: 0.82-0.88)",
                            attributes={
                                "metric": "ROC-AUC",
                                "value": "0.85",
                                "confidence_interval": "0.82-0.88",
                                "confidence_level": "95%"
                            }
                        ),
                        lx.data.Extraction(
                            extraction_class="baseline_comparison",
                            extraction_text="expert framework (manual assessment), structured-only model, and LLM-automated framework",
                            attributes={
                                "baseline_count": "3",
                                "comparison_type": "multiple baselines"
                            }
                        )
                    ]
                )
            ]
        }
    
    def _load_environment_config(self) -> Dict[str, Any]:
        """加载环境模块的LangExtract配置"""
        return {
            "prompt": textwrap.dedent("""
                Extract specific experimental environment information from medical papers. Follow these rules strictly:
                
                1. software_library: Extract specific software tools and library names
                2. hardware_resource: Extract descriptions of hardware resource requirements
                3. data_repository: Extract specific information about data storage and access
                4. code_availability: Extract specific descriptions of code availability
                5. compliance_requirement: Extract compliance and deployment requirements
                
                Use exact text for extraction, do not paraphrase. Provide meaningful attributes for each extraction.
                """),
            "examples": [
                lx.data.ExampleData(
                    text="We implemented the models using Python 3.8 with scikit-learn 1.0.2 and XGBoost 1.5.0. Training was performed on NVIDIA A100 GPU with 40GB memory. Code is available at GitHub: https://github.com/HealthRex/CDSS. The study was approved by Stanford IRB.",
                    extractions=[
                        lx.data.Extraction(
                            extraction_class="software_library",
                            extraction_text="Python 3.8 with scikit-learn 1.0.2 and XGBoost 1.5.0",
                            attributes={
                                "language": "Python",
                                "version": "3.8",
                                "libraries": "scikit-learn, XGBoost"
                            }
                        ),
                        lx.data.Extraction(
                            extraction_class="hardware_resource",
                            extraction_text="NVIDIA A100 GPU with 40GB memory",
                            attributes={
                                "gpu_type": "NVIDIA A100",
                                "memory": "40GB",
                                "resource_type": "GPU"
                            }
                        ),
                        lx.data.Extraction(
                            extraction_class="code_availability",
                            extraction_text="Code is available at GitHub: https://github.com/HealthRex/CDSS",
                            attributes={
                                "platform": "GitHub",
                                "url": "https://github.com/HealthRex/CDSS",
                                "access_type": "public"
                            }
                        ),
                        lx.data.Extraction(
                            extraction_class="compliance_requirement",
                            extraction_text="The study was approved by Stanford IRB",
                            attributes={
                                "approval_type": "IRB",
                                "institution": "Stanford"
                            }
                        )
                    ]
                )
            ]
        }
    
    def extract_paper_modules(self, paper_content: str, paper_id: str) -> Dict[str, Any]:
        """使用LangExtract提取论文的所有模块信息"""
        
        results = {
            "paper_id": paper_id,
            "extraction_metadata": {
                "timestamp": datetime.now().isoformat(),
                "method": "langextract_with_source_grounding",
                "model": "gpt-oss-20b"
            },
            "modules": {}
        }
        
        # 逐个提取每个模块
        for module_name, config in self.module_configs.items():
            # 模块提取重试机制，最多重试3次
            max_retries = 3
            extraction_result = None
            retry_errors = []
            
            for attempt in range(max_retries):
                try:
                    if attempt == 0:
                        logger.info(f"  提取{module_name}模块...")
                    else:
                        logger.info(f"  重试{module_name}模块... (尝试 {attempt + 1}/{max_retries})")
                    
                    # 使用LangExtract进行结构化提取
                    extraction_result = lx.extract(
                        text_or_documents=paper_content,
                        prompt_description=config["prompt"],
                        examples=config["examples"],
                        **self.extract_config
                    )
                    
                    # 检查提取是否成功
                    if extraction_result and hasattr(extraction_result, 'extractions') and extraction_result.extractions:
                        logger.info(f"    {module_name}模块提取成功 (尝试 {attempt + 1})")
                        break  # 成功，跳出重试循环
                    else:
                        error_msg = f"No valid extractions found (attempt {attempt + 1})"
                        retry_errors.append(error_msg)
                        logger.warning(f"    {module_name}模块提取失败: {error_msg}")
                        
                except Exception as e:
                    error_msg = f"API call failed (attempt {attempt + 1}): {str(e)}"
                    retry_errors.append(error_msg)
                    logger.error(f"    {module_name}模块提取异常: {error_msg}")
                    
                # 如果还有重试机会，稍作等待
                if attempt < max_retries - 1:
                    import time
                    time.sleep(1)  # 等待1秒再重试
            
            # 处理最终结果
            if extraction_result and hasattr(extraction_result, 'extractions') and extraction_result.extractions:
                results["modules"][module_name] = {
                    "extractions": [
                        {
                            "extraction_class": ext.extraction_class,
                            "extraction_text": ext.extraction_text,
                            "start_index": getattr(ext, 'start_index', None),
                            "end_index": getattr(ext, 'end_index', None),
                            "attributes": getattr(ext, 'attributes', {}),
                            "confidence": getattr(ext, 'confidence', None)
                        }
                        for ext in extraction_result.extractions
                    ],
                    "extraction_count": len(extraction_result.extractions),
                    "quality_score": self._calculate_quality_score(extraction_result),
                    "retry_attempts": len([e for e in retry_errors if e]) + 1  # 记录总尝试次数
                }
            else:
                # 所有重试都失败，使用默认值
                results["modules"][module_name] = {
                    "extractions": [],
                    "extraction_count": 0,
                    "quality_score": 0.0,
                    "error": f"All {max_retries} attempts failed",
                    "retry_errors": retry_errors,
                    "retry_attempts": max_retries
                }
        
        return results
    
    def _check_paper_already_extracted(self, papers_dir: str, paper_id: str) -> bool:
        """检查论文是否已经提取过，避免重复处理
        
        Args:
            papers_dir: 论文目录路径
            paper_id: 论文ID
            
        Returns:
            bool: True表示已提取过，False表示需要处理
        """
        paper_subdir = Path(papers_dir) / paper_id
        
        # 检查两个关键文件是否都存在
        json_file = paper_subdir / "mimic_langextract_dataset.json"
        html_file = paper_subdir / "mimic_langextract_dataset.html"
        
        return json_file.exists() and html_file.exists()
    
    def _preprocess_paper_content(self, content: str) -> str:
        """预处理论文内容，去除无关信息
        
        Args:
            content: 原始论文内容
            
        Returns:
            str: 处理后的论文内容
        """
        import re
        
        try:
            # 1. 去除Abstract之前的内容，如果没有Abstract则尝试Introduction
            # 优先寻找Abstract部分
            abstract_pattern = r'((?:abstract|ABSTRACT|Abstract)\s*:?\s*\n.*?)$'
            abstract_match = re.search(abstract_pattern, content, re.DOTALL | re.IGNORECASE)
            
            if abstract_match:
                content = abstract_match.group(1)
                logger.info("已保留Abstract及之后的内容")
            else:
                # 如果没有Abstract，尝试寻找Introduction
                intro_pattern = r'((?:introduction|INTRODUCTION|Introduction)\s*:?\s*\n.*?)$'
                intro_match = re.search(intro_pattern, content, re.DOTALL | re.IGNORECASE)
                
                if intro_match:
                    content = intro_match.group(1)
                    logger.info("已保留Introduction及之后的内容")
                else:
                    logger.info("未找到Abstract或Introduction标识，保持原内容")
            
            # 2. 去除References部分
            # 匹配References/REFERENCES/Bibliography等开始的部分到文末
            ref_patterns = [
                r'\n\s*(references|REFERENCES|References|bibliography|BIBLIOGRAPHY|Bibliography)\s*:?\s*\n.*$',
                r'\n\s*\d+\.\s*References\s*\n.*$',
                r'\n\s*参考文献\s*\n.*$'
            ]
            
            original_content_length = len(content)
            for pattern in ref_patterns:
                content = re.sub(pattern, '', content, flags=re.DOTALL | re.IGNORECASE)
            
            if len(content) != original_content_length:  # 检查是否有修改
                logger.info("已移除References部分")
            
            # 3. 去除所有URL链接
            url_patterns = [
                r'https?://[^\s\]\)]+',  # http/https链接
                r'www\.[^\s\]\)]+',      # www链接
                r'doi:[^\s\]\)]+',       # doi链接
                r'arxiv:[^\s\]\)]+',     # arxiv链接
            ]
            
            original_length = len(content)
            for pattern in url_patterns:
                content = re.sub(pattern, '[URL_REMOVED]', content, flags=re.IGNORECASE)
            
            if len(content) != original_length:
                logger.info("已移除URL链接")
            
            # 清理多余的空行
            content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)
            content = content.strip()
            
            return content
            
        except Exception as e:
            logger.warning(f"论文内容预处理失败: {e}，使用原始内容")
            return content
    
    def _process_single_paper(self, paper_item: tuple, papers_dir: str, total_papers: int) -> Dict[str, Any]:
        """处理单个论文的辅助方法，用于并行处理
        
        Args:
            paper_item: (paper_id, content) 元组
            papers_dir: 论文目录路径
            total_papers: 总论文数（用于进度显示）
            
        Returns:
            Dict[str, Any]: 包含论文ID和提取结果的字典
        """
        paper_id, content = paper_item
        
        try:
            # 检查是否已经提取过，避免重复处理
            if self._check_paper_already_extracted(papers_dir, paper_id):
                logger.info(f"跳过已处理论文: {paper_id} (输出文件已存在)")
                return {
                    "paper_id": paper_id,
                    "result": None,
                    "status": "skipped",
                    "reason": "已提取过，输出文件已存在"
                }
            
            logger.info(f"开始处理论文: {paper_id}")
            
            # 预处理论文内容，去除无关信息
            processed_content = self._preprocess_paper_content(content)
            logger.info(f"论文内容预处理完成: {paper_id}")
            
            # 提取论文模块信息
            paper_result = self.extract_paper_modules(processed_content, paper_id)
            
            # 为单个论文保存结果（这个操作应该是线程安全的，因为每个论文有独立的子目录）
            self._save_individual_paper_result(papers_dir, paper_id, paper_result)
            
            # 记录论文提取完成的进度日志
            successful_modules = sum(1 for module_data in paper_result.get('modules', {}).values() 
                                   if module_data.get('extraction_count', 0) > 0)
            total_modules = len(paper_result.get('modules', {}))
            total_extractions = sum(module_data.get('extraction_count', 0) 
                                  for module_data in paper_result.get('modules', {}).values())
            
            logger.info(f"✓ 论文提取完成: {paper_id} - 成功模块: {successful_modules}/{total_modules} - 总提取项: {total_extractions}")
            
            return {
                "paper_id": paper_id,
                "result": paper_result,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"处理论文 {paper_id} 失败: {e}")
            return {
                "paper_id": paper_id,
                "result": None,
                "status": "failed",
                "error": str(e)
            }
    
    def build_reproduction_dataset(self, papers_dir: str, output_file: str, max_papers: Optional[int] = None) -> Dict[str, Any]:
        """构建完整的复现数据集"""
        papers = self._load_markdown_papers(papers_dir)
        
        dataset = {
            "metadata": {
                "creation_date": datetime.now().isoformat(),
                "total_papers": len(papers),
                "extraction_method": "langextract_source_grounded",
                "api_endpoint": "http://100.82.33.121:11001/v1",
                "model": "gpt-oss-20b",
                "langextract_version": getattr(lx, '__version__', 'unknown')
            },
            "papers": {}
        }
        
        # 如果指定了最大处理数量，限制论文数量
        if max_papers and max_papers < len(papers):
            papers_items = list(papers.items())[:max_papers]
            papers = dict(papers_items)
            dataset["metadata"]["total_papers"] = len(papers)
            dataset["metadata"]["note"] = f"测试模式: 只处理前{max_papers}篇论文"
            logger.info(f"测试模式: 只处理前 {max_papers} 篇论文")
        
        # 统计需要处理的论文数（排除已处理的）
        papers_to_process = 0
        already_processed = 0
        
        for paper_id in papers.keys():
            if self._check_paper_already_extracted(papers_dir, paper_id):
                already_processed += 1
            else:
                papers_to_process += 1
        
        logger.info(f"发现 {len(papers)} 篇论文，已处理 {already_processed} 篇，待处理 {papers_to_process} 篇")
        logger.info(f"开始处理论文... (并行度: {self.doc_workers})")
        
        if papers_to_process == 0:
            logger.info("所有论文都已处理完成，无需重新提取")
            return dataset
        
        # 并行处理所有论文
        completed_count = 0
        paper_items = list(papers.items())
        
        with ThreadPoolExecutor(max_workers=self.doc_workers) as executor:
            # 提交所有任务
            future_to_paper = {
                executor.submit(self._process_single_paper, paper_item, papers_dir, len(papers)): paper_item[0]
                for paper_item in paper_items
            }
            
            # 处理完成的任务
            for future in as_completed(future_to_paper):
                completed_count += 1
                paper_id = future_to_paper[future]
                
                try:
                    result = future.result()
                    if result["status"] == "success":
                        dataset["papers"][paper_id] = result["result"]
                        logger.info(f"[{completed_count}/{len(papers)}] 完成论文: {paper_id}")
                    elif result["status"] == "skipped":
                        # 跳过的论文不计入失败，但需要记录日志
                        logger.info(f"[{completed_count}/{len(papers)}] 跳过论文: {paper_id} - {result.get('reason', '已处理')}")
                        # 跳过的论文可以选择不加入最终数据集或加入但标记为跳过
                        continue
                    else:
                        logger.error(f"[{completed_count}/{len(papers)}] 失败论文: {paper_id} - {result.get('error', '未知错误')}")
                        # 即使处理失败也要在数据集中记录
                        dataset["papers"][paper_id] = {
                            "paper_id": paper_id,
                            "extraction_metadata": {
                                "timestamp": datetime.now().isoformat(),
                                "method": "langextract_with_source_grounding",
                                "model": "gpt-oss-20b",
                                "error": result.get("error", "未知错误")
                            },
                            "modules": {}
                        }
                except Exception as e:
                    logger.error(f"[{completed_count}/{len(papers)}] 处理论文 {paper_id} 时发生异常: {e}")
                    # 记录异常情况
                    dataset["papers"][paper_id] = {
                        "paper_id": paper_id,
                        "extraction_metadata": {
                            "timestamp": datetime.now().isoformat(),
                            "method": "langextract_with_source_grounding", 
                            "model": "gpt-oss-20b",
                            "error": str(e)
                        },
                        "modules": {}
                    }
                
                # 定期保存全局进度（线程安全）
                if completed_count % 10 == 0:
                    with self.progress_lock:
                        self._save_progress(dataset, output_file)
        
        # 保存最终结果
        self._save_dataset(dataset, output_file)
        
        # 生成交互式HTML报告
        self._generate_html_report(dataset, output_file.replace('.json', '.html'))
        
        return dataset
    
    def _load_markdown_papers(self, papers_dir: str) -> Dict[str, str]:
        """加载markdown论文文件"""
        papers = {}
        papers_path = Path(papers_dir)
        
        if not papers_path.exists():
            raise FileNotFoundError(f"论文目录不存在: {papers_dir}")
        
        # 修改加载逻辑：从所有任务类型前缀的子目录中读取.md文件
        task_prefixes = ["PRED_", "CLAS_", "TIME_", "CORR_"]
        markdown_files = []
        valid_subdirs = []
        
        for subdir in papers_path.iterdir():
            if subdir.is_dir():
                # 检查是否以任何任务类型前缀开头
                has_task_prefix = any(subdir.name.startswith(prefix) for prefix in task_prefixes)
                if has_task_prefix:
                    valid_subdirs.append(subdir)
                    md_files = list(subdir.glob("*.md"))
                    markdown_files.extend(md_files)
        
        logger.info(f"发现 {len(valid_subdirs)} 个通过筛选的有效论文文件夹 (支持的任务类型前缀: {task_prefixes})")
        logger.info(f"有效文件夹列表: {[d.name for d in valid_subdirs[:5]]}")  # 显示前5个作为示例
        
        # 统计各类任务的数量
        task_counts = {prefix.rstrip('_').lower(): 0 for prefix in task_prefixes}
        for subdir in valid_subdirs:
            for prefix in task_prefixes:
                if subdir.name.startswith(prefix):
                    task_name = prefix.rstrip('_').lower()
                    task_counts[task_name] += 1
                    break
        logger.info(f"任务类型分布: {dict(task_counts)}")
        
        if not markdown_files:
            total_subdirs = len([d for d in papers_path.iterdir() if d.is_dir()])
            raise ValueError(f"在 {papers_dir} 目录中未找到有效的markdown文件 (总文件夹: {total_subdirs}, 有效文件夹: {len(valid_subdirs)}, 支持的前缀: {task_prefixes})")
        
        logger.info(f"发现 {len(markdown_files)} 个markdown文件")
        
        for file_path in markdown_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    papers[file_path.stem] = content
            except Exception as e:
                logger.warning(f"读取文件 {file_path} 失败: {e}")
        
        return papers
    
    def _calculate_quality_score(self, extraction_result) -> float:
        """计算提取质量分数"""
        if not extraction_result or not hasattr(extraction_result, 'extractions'):
            return 0.0
        
        if not extraction_result.extractions:
            return 0.0
        
        # 基于提取数量和属性丰富度计算质量分数
        total_score = 0.0
        for ext in extraction_result.extractions:
            score = 0.3  # 基础分数
            
            # 有源文本定位加分
            if hasattr(ext, 'start_index') and ext.start_index is not None:
                score += 0.2
            
            # 属性丰富度加分
            if ext.attributes and len(ext.attributes) > 0:
                score += min(0.3, len(ext.attributes) * 0.1)
            
            # 置信度加分
            if hasattr(ext, 'confidence') and ext.confidence:
                score += 0.2 * ext.confidence
            
            total_score += score
        
        return min(1.0, total_score / len(extraction_result.extractions))
    
    def _save_progress(self, dataset: Dict[str, Any], output_file: str):
        """保存处理进度"""
        try:
            progress_file = output_file.replace('.json', '_progress.json')
            with open(progress_file, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, ensure_ascii=False, indent=2)
            logger.info(f"进度已保存至: {progress_file}")
        except Exception as e:
            logger.error(f"保存进度失败: {e}")
    
    def _save_individual_paper_result(self, papers_dir: str, paper_id: str, paper_result: Dict[str, Any]):
        """为单个论文保存提取结果到其对应的子文件夹"""
        try:
            # 构建论文子文件夹路径
            paper_subdir = Path(papers_dir) / paper_id
            if not paper_subdir.exists():
                logger.warning(f"论文子文件夹不存在: {paper_subdir}")
                return
            
            # 准备单个论文的数据集格式
            individual_dataset = {
                "metadata": {
                    "creation_date": datetime.now().isoformat(),
                    "total_papers": 1,
                    "extraction_method": "langextract_source_grounded",
                    "api_endpoint": "http://100.82.33.121:11001/v1",
                    "model": "gpt-oss-20b",
                    "langextract_version": getattr(lx, '__version__', 'unknown'),
                    "paper_id": paper_id
                },
                "paper": paper_result  # 注意：这里是单个论文，所以用"paper"而不是"papers"
            }
            
            # 保存JSON文件
            json_file = paper_subdir / "mimic_langextract_dataset.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(individual_dataset, f, ensure_ascii=False, indent=2)
            
            # 生成HTML报告
            html_file = paper_subdir / "mimic_langextract_dataset.html"
            self._generate_individual_html_report(individual_dataset, html_file)
            
            logger.info(f"已保存论文 {paper_id} 的结果到: {paper_subdir}")
            
        except Exception as e:
            logger.error(f"保存单个论文结果失败 ({paper_id}): {e}")
    
    def _save_dataset(self, dataset: Dict[str, Any], output_file: str):
        """保存最终数据集"""
        try:
            # 确保输出目录存在
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, ensure_ascii=False, indent=2)
            logger.info(f"数据集已保存至: {output_file}")
        except Exception as e:
            logger.error(f"保存数据集失败: {e}")
            raise
    
    def _generate_html_report(self, dataset: Dict[str, Any], output_file: str):
        """生成LangExtract风格的交互式HTML报告"""
        try:
            # 合并所有提取结果用于可视化
            all_extractions = []
            for paper_id, paper_data in dataset["papers"].items():
                for module_name, module_data in paper_data.get("modules", {}).items():
                    all_extractions.extend(module_data.get("extractions", []))
            
            # 基础HTML模板（简化版可视化）
            html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>MIMIC复现数据集 - LangExtract报告</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background: #f0f8ff; padding: 20px; border-radius: 5px; }}
        .stats {{ display: flex; gap: 20px; margin: 20px 0; }}
        .stat-card {{ background: #e6f3ff; padding: 15px; border-radius: 5px; }}
        .extraction {{ border: 1px solid #ddd; margin: 10px 0; padding: 15px; border-radius: 5px; }}
        .class-tag {{ background: #007acc; color: white; padding: 3px 8px; border-radius: 3px; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>MIMIC复现数据集 - LangExtract提取报告</h1>
        <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>总论文数: {dataset['metadata']['total_papers']}</p>
        <p>提取方法: {dataset['metadata']['extraction_method']}</p>
    </div>
    
    <div class="stats">
        <div class="stat-card">
            <h3>提取统计</h3>
            <p>总提取项: {len(all_extractions)}</p>
            <p>平均每篇: {len(all_extractions)/dataset['metadata']['total_papers']:.1f}</p>
        </div>
        <div class="stat-card">
            <h3>成功率</h3>
            <p>处理成功: {len([p for p in dataset['papers'].values() if any(m.get('extraction_count', 0) > 0 for m in p.get('modules', {}).values())])}/{dataset['metadata']['total_papers']}</p>
        </div>
    </div>
    
    <div class="extractions">
        <h2>提取结果示例</h2>
"""
            
            # 添加前20个提取结果作为示例
            for i, ext in enumerate(all_extractions[:20]):
                html_content += f"""
        <div class="extraction">
            <span class="class-tag">{ext.get('extraction_class', 'unknown')}</span>
            <p><strong>提取文本:</strong> "{ext.get('extraction_text', 'N/A')}"</p>
            <p><strong>属性:</strong> {ext.get('attributes', {})}</p>
            <p><strong>置信度:</strong> {ext.get('confidence', 'N/A')}</p>
        </div>
"""
            
            html_content += """
    </div>
</body>
</html>
"""
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
                
            logger.info(f"交互式报告已生成: {output_file}")
            
        except Exception as e:
            logger.error(f"HTML报告生成失败: {e}")
    
    def _generate_individual_html_report(self, individual_dataset: Dict[str, Any], output_file: Path):
        """生成单个论文的LangExtract风格交互式HTML报告"""
        try:
            # 从单个论文数据中提取所有提取结果
            paper_data = individual_dataset["paper"]
            all_extractions = []
            for module_name, module_data in paper_data.get("modules", {}).items():
                for ext in module_data.get("extractions", []):
                    ext["module"] = module_name  # 添加模块标识
                    all_extractions.append(ext)
            
            # 计算统计信息
            successful_modules = len([
                module for module in paper_data.get("modules", {}).values()
                if module.get("extraction_count", 0) > 0
            ])
            total_modules = len(paper_data.get("modules", {}))
            
            # 生成HTML内容
            html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>{individual_dataset['metadata']['paper_id']} - LangExtract提取报告</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f9f9f9; }}
        .header {{ background: #e3f2fd; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .stats {{ display: flex; gap: 20px; margin: 20px 0; }}
        .stat-card {{ background: #ffffff; padding: 15px; border-radius: 8px; flex: 1; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .extraction {{ border: 1px solid #e0e0e0; margin: 15px 0; padding: 15px; border-radius: 8px; background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        .class-tag {{ background: #1976d2; color: white; padding: 4px 10px; border-radius: 12px; font-size: 12px; margin-right: 10px; }}
        .module-tag {{ background: #388e3c; color: white; padding: 2px 8px; border-radius: 10px; font-size: 11px; margin-left: 10px; }}
        .attributes {{ background: #f5f5f5; padding: 10px; border-radius: 4px; margin-top: 10px; font-size: 13px; }}
        .no-extractions {{ text-align: center; color: #666; padding: 40px; background: #f0f0f0; border-radius: 8px; }}
        h1 {{ color: #1565c0; margin: 0; }}
        h2 {{ color: #424242; }}
        h3 {{ color: #1976d2; margin: 0; }}
        .meta-info {{ color: #666; font-size: 14px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>MIMIC论文信息提取报告</h1>
        <h2>{individual_dataset['metadata']['paper_id']}</h2>
        <div class="meta-info">
            <p><strong>生成时间:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>提取方法:</strong> {individual_dataset['metadata']['extraction_method']}</p>
            <p><strong>模型:</strong> {individual_dataset['metadata']['model']}</p>
        </div>
    </div>
    
    <div class="stats">
        <div class="stat-card">
            <h3>提取统计</h3>
            <p><strong>总提取项:</strong> {len(all_extractions)}</p>
            <p><strong>成功模块:</strong> {successful_modules}/{total_modules}</p>
        </div>
        <div class="stat-card">
            <h3>模块分布</h3>
"""
            
            # 添加每个模块的统计信息
            for module_name, module_data in paper_data.get("modules", {}).items():
                extraction_count = module_data.get("extraction_count", 0)
                html_content += f"            <p><strong>{module_name}:</strong> {extraction_count} 项</p>\n"
            
            html_content += """
        </div>
    </div>
    
    <div class="extractions">
        <h2>详细提取结果</h2>
"""
            
            if all_extractions:
                # 按模块分组显示提取结果
                for module_name in ["data", "model", "training", "evaluation", "environment"]:
                    module_extractions = [ext for ext in all_extractions if ext.get("module") == module_name]
                    if module_extractions:
                        html_content += f"""        <h3>{module_name.title()} 模块 ({len(module_extractions)} 项)</h3>\n"""
                        
                        for ext in module_extractions:
                            confidence_text = f" (置信度: {ext.get('confidence', 'N/A')})" if ext.get('confidence') else ""
                            html_content += f"""
        <div class="extraction">
            <span class="class-tag">{ext.get('extraction_class', 'unknown')}</span>
            <span class="module-tag">{module_name}</span>
            <p><strong>提取文本:</strong> "{ext.get('extraction_text', 'N/A')}"</p>
"""
                            # 添加属性信息
                            attributes = ext.get('attributes', {})
                            if attributes:
                                html_content += f"""            <div class="attributes">
                <strong>属性:</strong> """
                                for key, value in attributes.items():
                                    html_content += f"<span><strong>{key}:</strong> {value}</span> &nbsp;&nbsp; "
                                html_content += """
            </div>"""
                            
                            # 添加位置信息
                            if ext.get('start_index') is not None and ext.get('end_index') is not None:
                                html_content += f"""            <p class="meta-info">位置: {ext.get('start_index')}-{ext.get('end_index')}{confidence_text}</p>"""
                            
                            html_content += """        </div>
"""
            else:
                html_content += """
        <div class="no-extractions">
            <p>未找到任何提取结果</p>
            <p>可能的原因：模型无法识别相关信息，或者文本内容不包含目标信息类型</p>
        </div>
"""
            
            html_content += """
    </div>
</body>
</html>
"""
            
            # 写入HTML文件
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
                
            logger.info(f"单个论文HTML报告已生成: {output_file}")
            
        except Exception as e:
            logger.error(f"单个论文HTML报告生成失败: {e}")