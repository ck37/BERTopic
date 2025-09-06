"""
Specific test to reproduce the exact bug scenario from GitHub issue #2384.
This test tries to create the exact conditions where topic_sizes_ is not updated correctly.
"""

import numpy as np
from bertopic import BERTopic


def test_exact_bug_scenario():
    """
    Test the exact scenario described in the GitHub issue.
    
    The bug occurs when:
    1. Zero-shot topic modeling assigns ALL documents to predefined topics
    2. No documents remain for clustering (len(documents) == 0 after zero-shot)
    3. nr_topics is used for topic reduction
    4. topic_sizes_ is not properly updated
    """
    print("Testing exact bug scenario...")
    
    # Create documents that will definitely match zero-shot topics
    docs = [
        "This is about machine learning and artificial intelligence",
        "Deep learning neural networks are powerful",
        "Python programming for data science",
        "Machine learning algorithms and models",
        "Artificial intelligence and deep learning",
        "Data science with Python programming",
        "Neural networks and machine learning",
        "Programming in Python for AI",
        "Deep learning models and algorithms",
        "Artificial intelligence programming"
    ]
    
    # Zero-shot topics with very broad matching
    zeroshot_topics = ["Technology and Programming"]
    
    # Create model that should assign ALL docs to zero-shot topics
    topic_model = BERTopic(
        zeroshot_topic_list=zeroshot_topics,
        zeroshot_min_similarity=0.1,  # Very low threshold
        nr_topics=2,  # Must exceed number of zero-shot topics (1)
        min_topic_size=2,
        verbose=True  # Enable verbose to see what's happening
    )
    
    print(f"Documents: {len(docs)}")
    print(f"Zero-shot topics: {zeroshot_topics}")
    
    # Fit the model
    topics, probs = topic_model.fit_transform(docs)
    
    print(f"Assigned topics: {topics}")
    print(f"Topic sizes from topic_sizes_: {topic_model.topic_sizes_}")
    
    # Count actual topic assignments
    actual_counts = {}
    for topic in topics:
        actual_counts[topic] = actual_counts.get(topic, 0) + 1
    
    print(f"Actual topic counts: {actual_counts}")
    
    # Check for the bug
    total_in_sizes = sum(topic_model.topic_sizes_.values())
    total_actual = len(docs)
    
    print(f"Total docs in topic_sizes_: {total_in_sizes}")
    print(f"Total actual docs: {total_actual}")
    
    if total_in_sizes != total_actual:
        print(f"🐛 BUG DETECTED: topic_sizes_ total ({total_in_sizes}) != actual docs ({total_actual})")
        return False
    
    # Check individual topic counts
    for topic_id, expected_count in actual_counts.items():
        if topic_id not in topic_model.topic_sizes_:
            print(f"🐛 BUG DETECTED: Topic {topic_id} missing from topic_sizes_")
            return False
        
        actual_size = topic_model.topic_sizes_[topic_id]
        if actual_size != expected_count:
            print(f"🐛 BUG DETECTED: Topic {topic_id} size mismatch - expected {expected_count}, got {actual_size}")
            return False
    
    print("✅ No bug detected - topic_sizes_ is correctly updated")
    return True


def test_edge_case_all_zeroshot_no_clustering():
    """
    Test the specific edge case where all documents are assigned to zero-shot
    and no clustering occurs.
    """
    print("\nTesting edge case: all zero-shot, no clustering...")
    
    # Use more documents to avoid UMAP issues with small datasets
    docs = [
        "Technology is advancing rapidly",
        "Software development is important", 
        "Programming languages are evolving",
        "Computer science research continues",
        "Digital transformation is happening",
        "Innovation in technology sector",
        "Software engineering best practices",
        "Modern programming techniques",
        "Computer systems and architecture",
        "Digital solutions and platforms",
        "Technology trends and developments",
        "Software design patterns",
        "Programming paradigms evolution",
        "Computing infrastructure advances",
        "Digital innovation strategies"
    ]
    
    # Single zero-shot topic that should capture everything
    zeroshot_topics = ["Technology"]
    
    # Configure UMAP for small datasets
    from umap import UMAP
    umap_model = UMAP(n_neighbors=5, n_components=2, min_dist=0.0, metric='cosine', random_state=42)
    
    topic_model = BERTopic(
        zeroshot_topic_list=zeroshot_topics,
        zeroshot_min_similarity=0.05,  # Very low to ensure all docs match
        nr_topics=2,  # Must exceed number of zero-shot topics (1)
        min_topic_size=1,
        umap_model=umap_model,
        verbose=True
    )
    
    topics, probs = topic_model.fit_transform(docs)
    
    print(f"Topics assigned: {set(topics)}")
    print(f"Topic sizes: {topic_model.topic_sizes_}")
    
    # Verify all documents are accounted for
    total_in_sizes = sum(topic_model.topic_sizes_.values())
    if total_in_sizes != len(docs):
        print(f"🐛 BUG: Missing documents - {len(docs) - total_in_sizes} docs not in topic_sizes_")
        return False
    
    print("✅ Edge case handled correctly")
    return True


def test_debug_zero_shot_flow():
    """
    Debug the zero-shot flow to understand what's happening internally.
    """
    print("\nDebugging zero-shot flow...")
    
    # Use more documents to avoid UMAP issues
    docs = [
        "AI and machine learning research",
        "Deep learning neural networks", 
        "Neural network architectures",
        "Machine learning algorithms",
        "Artificial intelligence systems",
        "Deep learning models training",
        "Neural network optimization",
        "Machine learning applications",
        "AI research and development",
        "Deep learning frameworks"
    ]
    zeroshot_topics = ["Artificial Intelligence"]
    
    # Configure UMAP for small datasets
    from umap import UMAP
    umap_model = UMAP(n_neighbors=5, n_components=2, min_dist=0.0, metric='cosine', random_state=42)
    
    topic_model = BERTopic(
        zeroshot_topic_list=zeroshot_topics,
        zeroshot_min_similarity=0.1,
        nr_topics=2,  # Must exceed number of zero-shot topics (1)
        min_topic_size=1,
        umap_model=umap_model,
        verbose=True
    )
    
    print("Before fit_transform:")
    print(f"  topic_sizes_: {topic_model.topic_sizes_}")
    
    topics, probs = topic_model.fit_transform(docs)
    
    print("After fit_transform:")
    print(f"  topics: {topics}")
    print(f"  topic_sizes_: {topic_model.topic_sizes_}")
    print(f"  topics_: {topic_model.topics_}")
    
    # Check topic info consistency
    topic_info = topic_model.get_topic_info()
    print(f"  topic_info counts: {dict(zip(topic_info.Topic, topic_info.Count))}")
    
    return True


if __name__ == "__main__":
    print("Running specific bug reproduction tests...\n")
    
    success1 = test_exact_bug_scenario()
    success2 = test_edge_case_all_zeroshot_no_clustering() 
    success3 = test_debug_zero_shot_flow()
    
    if success1 and success2 and success3:
        print("\n🎉 All tests passed - bug may already be fixed or conditions not met")
    else:
        print("\n❌ Some tests failed - bug reproduction successful")
