# Advanced Reranking Techniques for RAG Systems

A comprehensive implementation of state-of-the-art reranking methods for Retrieval-Augmented Generation (RAG) systems, featuring semantic reranking, cross-encoder reranking, and LLM-based reranking approaches.

## 🚀 Overview

This repository demonstrates three advanced reranking techniques that significantly improve the quality of document retrieval in RAG systems:

1. **Semantic Reranking (Bi-Encoder)** - Fast, efficient semantic similarity-based reranking
2. **Cross-Encoder Reranking** - Superior accuracy through joint query-document processing  
3. **LLM-Based Reranking** - Leverages large language models for nuanced relevance assessment

## 📊 Reranking Methods Comparison

| Method | Accuracy | Speed | Cost | Best Use Case |
|--------|----------|-------|------|---------------|
| **Semantic (Bi-Encoder)** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | High-volume, real-time applications |
| **Cross-Encoder** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | Quality-critical applications |
| **LLM-Based** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ | Complex queries, domain expertise |

## 🎯 Key Features

- **Multiple Reranking Strategies**: Implement and compare different reranking approaches
- **Performance Analysis**: Comprehensive timing and quality metrics
- **Interactive Demos**: Test different reranking methods with custom queries
- **Google Gemini Integration**: Advanced answer generation with detailed explanations
- **Visualization Tools**: Compare reranking effectiveness across methods
- **Production-Ready**: Scalable implementations with performance optimizations

## 🛠️ Installation

### Prerequisites

```bash
Python 3.8+
```

### Required Dependencies

```bash
pip install sentence-transformers faiss-cpu numpy pandas google-generativeai python-dotenv scikit-learn transformers torch matplotlib seaborn
```

### Environment Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd rag-reranking-system
```

2. **Set up environment variables**
```bash
# Create .env file
echo "GEMINI_API_KEY=your-gemini-api-key-here" > .env
```

3. **Get your Gemini API key**
   - Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Generate and copy your API key
   - Add it to your `.env` file

## 📚 Quick Start

### 1. Semantic Reranking (Bi-Encoder)

```python
from semantic_reranking import SemanticReranker, RAGSystem
import google.generativeai as genai

# Initialize components
reranker = SemanticReranker(model_name='all-MiniLM-L6-v2')
reranker.build_index(documents)

# Initialize RAG system
gemini_model = genai.GenerativeModel('gemini-1.5-flash')
rag_system = RAGSystem(reranker, gemini_model)

# Process query
result = rag_system.process_query(
    "What do we know about life on Mars?", 
    retrieve_k=8, 
    rerank_k=3
)
```

### 2. Cross-Encoder Reranking

```python
from cross_encoder_reranking import CrossEncoderReranker, AdvancedRAGSystem

# Initialize cross-encoder system
cross_encoder_reranker = CrossEncoderReranker(
    bi_encoder_model='all-MiniLM-L6-v2',
    cross_encoder_model='cross-encoder/ms-marco-MiniLM-L-6-v2'
)
cross_encoder_reranker.build_index(documents)

# Initialize advanced RAG system
advanced_rag = AdvancedRAGSystem(cross_encoder_reranker, gemini_model)

# Process with comparison
result = advanced_rag.process_query_with_comparison(
    "How do space agencies search for signs of life beyond Earth?",
    retrieve_k=12,
    rerank_k=4
)
```

### 3. LLM-Based Reranking

```python
import google.generativeai as genai

class LLMReranker:
    def __init__(self, model_name='gemini-1.5-flash'):
        self.model = genai.GenerativeModel(model_name)
    
    def llm_rerank(self, query: str, documents: List[Dict], top_k: int = 5):
        """Rerank documents using LLM for relevance assessment"""
        
        # Prepare documents for LLM evaluation
        doc_summaries = []
        for i, doc in enumerate(documents):
            doc_summaries.append(f"{i+1}. {doc['title']}: {doc['content'][:200]}...")
        
        prompt = f"""
        Given the query: "{query}"
        
        Rank the following documents by relevance (1 = most relevant):
        
        {chr(10).join(doc_summaries)}
        
        Return only the ranking as numbers separated by commas (e.g., 3,1,4,2,5).
        Consider semantic relevance, query intent, and content quality.
        """
        
        try:
            response = self.model.generate_content(prompt)
            rankings = [int(x.strip()) for x in response.text.strip().split(',')]
            
            # Reorder documents based on LLM ranking
            reranked_docs = []
            for rank in rankings[:top_k]:
                if 1 <= rank <= len(documents):
                    reranked_docs.append(documents[rank-1])
            
            return reranked_docs
            
        except Exception as e:
            print(f"LLM reranking failed: {e}")
            return documents[:top_k]

# Usage
llm_reranker = LLMReranker()
reranked_docs = llm_reranker.llm_rerank(query, retrieved_docs, top_k=5)
```

## 📖 Detailed Usage Examples

### Running the Notebooks

1. **Semantic Reranking Demo**
```bash
jupyter notebook "Semantic Reranking.ipynb"
```

2. **Cross-Encoder Reranking Demo**
```bash
jupyter notebook "Cross-Encoder Reranking.ipynb"
```

### Interactive Testing

```python
def interactive_comparison_demo():
    """Compare all three reranking methods interactively"""
    
    print("🚀 Multi-Method Reranking Comparison")
    
    while True:
        query = input("🔍 Your question: ").strip()
        if query.lower() == 'quit':
            break
            
        print("\n" + "="*80)
        print(f"Query: {query}")
        print("="*80)
        
        # Method 1: Semantic Reranking
        print("\n🔵 SEMANTIC RERANKING (Bi-Encoder)")
        semantic_result = semantic_rag.process_query(query, retrieve_k=10, rerank_k=5)
        display_top_results("Semantic", semantic_result)
        
        # Method 2: Cross-Encoder Reranking  
        print("\n🟢 CROSS-ENCODER RERANKING")
        cross_result = cross_encoder_rag.process_query_with_comparison(query, retrieve_k=10, rerank_k=5)
        display_top_results("Cross-Encoder", cross_result)
        
        # Method 3: LLM-Based Reranking
        print("\n🟡 LLM-BASED RERANKING")
        llm_result = llm_rag.process_query(query, retrieve_k=10, rerank_k=5)
        display_top_results("LLM", llm_result)
        
        print("\n" + "="*80)

# Run the interactive demo
interactive_comparison_demo()
```

## 🔬 Performance Analysis

### Comprehensive Benchmarking

```python
def benchmark_all_methods(test_queries, documents):
    """Benchmark all reranking methods"""
    
    results = {
        'semantic': {'times': [], 'relevance_scores': []},
        'cross_encoder': {'times': [], 'relevance_scores': []},
        'llm': {'times': [], 'relevance_scores': []}
    }
    
    for query in test_queries:
        print(f"Benchmarking: {query}")
        
        # Semantic reranking
        start_time = time.time()
        semantic_result = semantic_reranker.semantic_rerank(query, documents, top_k=5)
        semantic_time = time.time() - start_time
        results['semantic']['times'].append(semantic_time)
        
        # Cross-encoder reranking
        start_time = time.time()
        cross_result = cross_encoder_reranker.cross_encoder_rerank(query, documents, top_k=5)
        cross_time = time.time() - start_time
        results['cross_encoder']['times'].append(cross_time)
        
        # LLM reranking
        start_time = time.time()
        llm_result = llm_reranker.llm_rerank(query, documents, top_k=5)
        llm_time = time.time() - start_time
        results['llm']['times'].append(llm_time)
    
    # Display results
    print("\n📊 PERFORMANCE COMPARISON")
    print("-" * 50)
    print(f"{'Method':<15} {'Avg Time (s)':<12} {'Relative Speed':<15}")
    print("-" * 50)
    
    semantic_avg = np.mean(results['semantic']['times'])
    cross_avg = np.mean(results['cross_encoder']['times'])
    llm_avg = np.mean(results['llm']['times'])
    
    print(f"{'Semantic':<15} {semantic_avg:<12.3f} {'1.0x':<15}")
    print(f"{'Cross-Encoder':<15} {cross_avg:<12.3f} {f'{cross_avg/semantic_avg:.1f}x':<15}")
    print(f"{'LLM-Based':<15} {llm_avg:<12.3f} {f'{llm_avg/semantic_avg:.1f}x':<15}")
    
    return results
```

## 🎯 Method Selection Guide

### When to Use Each Method

#### Semantic Reranking (Bi-Encoder)
**Best for:**
- High-volume applications (>1000 queries/hour)
- Real-time systems requiring <100ms response
- Cost-sensitive deployments
- General domain applications

**Characteristics:**
- ⚡ Fastest execution
- 💰 Lowest cost
- 🎯 Good baseline accuracy
- 📈 Highly scalable

#### Cross-Encoder Reranking
**Best for:**
- Quality-critical applications
- Complex multi-aspect queries
- Domain-specific knowledge bases
- Medium-volume applications (<500 queries/hour)

**Characteristics:**
- 🎯 Highest accuracy for relevance
- ⚖️ Balanced cost/performance
- 🧠 Better context understanding
- 🔄 Requires moderate compute resources

#### LLM-Based Reranking
**Best for:**
- Complex reasoning tasks
- Domain expertise requirements
- Multi-criteria evaluation
- Low-volume, high-value queries

**Characteristics:**
- 🧠 Best semantic understanding
- 💰 Highest cost
- 🐌 Slower execution
- 🎭 Most flexible and adaptable

## 📈 Advanced Features

### Hybrid Reranking Strategy

```python
class HybridReranker:
    def __init__(self, semantic_weight=0.3, cross_encoder_weight=0.5, llm_weight=0.2):
        self.semantic_reranker = SemanticReranker()
        self.cross_encoder_reranker = CrossEncoderReranker()
        self.llm_reranker = LLMReranker()
        
        self.weights = {
            'semantic': semantic_weight,
            'cross_encoder': cross_encoder_weight,
            'llm': llm_weight
        }
    
    def hybrid_rerank(self, query: str, documents: List[Dict], top_k: int = 5):
        """Combine multiple reranking signals"""
        
        # Get scores from each method
        semantic_scores = self.semantic_reranker.get_scores(query, documents)
        cross_scores = self.cross_encoder_reranker.get_scores(query, documents)
        llm_scores = self.llm_reranker.get_scores(query, documents)
        
        # Normalize scores
        semantic_scores = self._normalize_scores(semantic_scores)
        cross_scores = self._normalize_scores(cross_scores)
        llm_scores = self._normalize_scores(llm_scores)
        
        # Combine with weights
        final_scores = []
        for i in range(len(documents)):
            combined_score = (
                semantic_scores[i] * self.weights['semantic'] +
                cross_scores[i] * self.weights['cross_encoder'] +
                llm_scores[i] * self.weights['llm']
            )
            final_scores.append((documents[i], combined_score))
        
        # Sort by combined score
        final_scores.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in final_scores[:top_k]]
```

### Query-Adaptive Reranking

```python
class AdaptiveReranker:
    def __init__(self):
        self.semantic_reranker = SemanticReranker()
        self.cross_encoder_reranker = CrossEncoderReranker()
        self.llm_reranker = LLMReranker()
    
    def classify_query_complexity(self, query: str) -> str:
        """Classify query to choose appropriate reranking method"""
        
        if len(query.split()) <= 5 and not any(word in query.lower() for word in ['compare', 'analyze', 'explain why']):
            return 'simple'
        elif any(word in query.lower() for word in ['compare', 'contrast', 'relationship', 'difference']):
            return 'complex'
        else:
            return 'medium'
    
    def adaptive_rerank(self, query: str, documents: List[Dict], top_k: int = 5):
        """Choose reranking method based on query complexity"""
        
        complexity = self.classify_query_complexity(query)
        
        if complexity == 'simple':
            return self.semantic_reranker.semantic_rerank(query, documents, top_k)
        elif complexity == 'medium':
            return self.cross_encoder_reranker.cross_encoder_rerank(query, documents, top_k)
        else:  # complex
            return self.llm_reranker.llm_rerank(query, documents, top_k)
```

## 🔧 Configuration Options

### Model Configuration

```python
# Semantic Reranking Models
SEMANTIC_MODELS = {
    'fast': 'all-MiniLM-L6-v2',           # Fastest, good quality
    'balanced': 'all-mpnet-base-v2',       # Balanced speed/quality
    'quality': 'all-MiniLM-L12-v2'        # Best quality, slower
}

# Cross-Encoder Models
CROSS_ENCODER_MODELS = {
    'fast': 'cross-encoder/ms-marco-TinyBERT-L-2-v2',
    'balanced': 'cross-encoder/ms-marco-MiniLM-L-6-v2',
    'quality': 'cross-encoder/ms-marco-electra-base'
}

# LLM Models
LLM_MODELS = {
    'fast': 'gemini-1.5-flash',
    'balanced': 'gemini-1.5-pro',
    'quality': 'gemini-1.5-pro-latest'
}
```

### Production Configuration

```python
PRODUCTION_CONFIG = {
    'vector_store': 'pinecone',  # or 'weaviate', 'qdrant'
    'embedding_cache': True,
    'result_cache_ttl': 3600,   # 1 hour
    'max_concurrent_requests': 10,
    'timeout_seconds': 30,
    'fallback_enabled': True
}
```

## 📊 Monitoring and Evaluation

### Quality Metrics

```python
def evaluate_reranking_quality(ground_truth, reranked_results):
    """Evaluate reranking quality using multiple metrics"""
    
    metrics = {}
    
    # NDCG (Normalized Discounted Cumulative Gain)
    metrics['ndcg@5'] = calculate_ndcg(ground_truth, reranked_results, k=5)
    
    # MRR (Mean Reciprocal Rank)
    metrics['mrr'] = calculate_mrr(ground_truth, reranked_results)
    
    # Precision@K
    for k in [1, 3, 5]:
        metrics[f'precision@{k}'] = calculate_precision_at_k(ground_truth, reranked_results, k)
    
    # MAP (Mean Average Precision)
    metrics['map'] = calculate_map(ground_truth, reranked_results)
    
    return metrics
```

### A/B Testing Framework

```python
class RetrievalABTest:
    def __init__(self, control_method, test_method):
        self.control_method = control_method
        self.test_method = test_method
        self.results = {'control': [], 'test': []}
    
    def run_test(self, queries, user_feedback_fn):
        """Run A/B test comparing reranking methods"""
        
        for query in queries:
            # Randomly assign to control or test
            if random.random() < 0.5:
                method = 'control'
                result = self.control_method.process_query(query)
            else:
                method = 'test' 
                result = self.test_method.process_query(query)
            
            # Collect user feedback
            feedback = user_feedback_fn(query, result)
            self.results[method].append(feedback)
        
        return self.analyze_results()
    
    def analyze_results(self):
        """Analyze A/B test results"""
        control_satisfaction = np.mean(self.results['control'])
        test_satisfaction = np.mean(self.results['test'])
        
        # Statistical significance test
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(self.results['control'], self.results['test'])
        
        return {
            'control_mean': control_satisfaction,
            'test_mean': test_satisfaction,
            'improvement': (test_satisfaction - control_satisfaction) / control_satisfaction * 100,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
```

## 🚀 Deployment Guide

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### FastAPI Service

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="Advanced RAG Reranking Service")

class QueryRequest(BaseModel):
    query: str
    method: str = 'cross_encoder'  # 'semantic', 'cross_encoder', 'llm', 'hybrid'
    retrieve_k: int = 10
    rerank_k: int = 5

class RetrievalResult(BaseModel):
    documents: List[dict]
    scores: List[float]
    method_used: str
    processing_time: float

@app.post("/rerank", response_model=RetrievalResult)
async def rerank_documents(request: QueryRequest):
    """Rerank documents using specified method"""
    
    try:
        start_time = time.time()
        
        if request.method == 'semantic':
            result = semantic_rag.process_query(request.query, request.retrieve_k, request.rerank_k)
        elif request.method == 'cross_encoder':
            result = cross_encoder_rag.process_query(request.query, request.retrieve_k, request.rerank_k)
        elif request.method == 'llm':
            result = llm_rag.process_query(request.query, request.retrieve_k, request.rerank_k)
        elif request.method == 'hybrid':
            result = hybrid_rag.process_query(request.query, request.retrieve_k, request.rerank_k)
        else:
            raise HTTPException(status_code=400, detail="Invalid reranking method")
        
        processing_time = time.time() - start_time
        
        return RetrievalResult(
            documents=result['context_docs'],
            scores=result.get('scores', []),
            method_used=request.method,
            processing_time=processing_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
```

## 📚 References and Further Reading

### Academic Papers
- [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084)
- [ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT](https://arxiv.org/abs/2004.12832)
- [Cross-Encoders for Information Retrieval](https://arxiv.org/abs/1910.10683)

### Model Documentation
- [Sentence Transformers](https://www.sbert.net/)
- [Google Gemini API](https://ai.google.dev/)
- [FAISS Documentation](https://faiss.ai/)

### Best Practices
- [RAG Evaluation Best Practices](https://docs.llamaindex.ai/en/stable/examples/evaluation/)
- [Vector Database Comparison](https://superlinked.com/vector-db-comparison)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-reranking-method`)
3. Commit your changes (`git commit -am 'Add new reranking method'`)
4. Push to the branch (`git push origin feature/new-reranking-method`)
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙋‍♂️ Support

- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/discussions)
- **Email**: your-email@example.com

---

**Happy Reranking! 🚀**
