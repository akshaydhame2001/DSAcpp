#include <bits/stdc++.h>
using namespace std;

void explainPriorityQueue()
{
    // Max-Heap (default behavior)
    priority_queue<int> maxHeap;

    // Insert elements into the max-heap
    maxHeap.push(5);     // {5}
    maxHeap.push(2);     // {5, 2}
    maxHeap.push(8);     // {8, 5, 2}
    maxHeap.emplace(10); // {10, 8, 5, 2}

    // Print the largest element
    cout << "Max-Heap Top: " << maxHeap.top() << endl; // Output: 10

    // Remove the largest element
    maxHeap.pop(); // {8, 5, 2}

    // Print the new largest element
    cout << "Max-Heap Top after pop: " << maxHeap.top() << endl; // Output: 8

    // Min-Heap (using greater<int>)
    priority_queue<int, vector<int>, greater<int>> minHeap;

    // Insert elements into the min-heap
    minHeap.push(5);     // {5}
    minHeap.push(2);     // {2, 5}
    minHeap.push(8);     // {2, 5, 8}
    minHeap.emplace(10); // {2, 5, 8, 10}

    // Print the smallest element
    cout << "Min-Heap Top: " << minHeap.top() << endl; // Output: 2
}

int main()
{
    explainPriorityQueue();
    return 0;
}

/*

Max-Heap Top: 10
Max-Heap Top after pop: 8
Min-Heap Top: 2

*/