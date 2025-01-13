#include <bits/stdc++.h>
using namespace std;

void explainQueue()
{
    // Declare a queue of integers
    queue<int> q;

    // Push elements into the queue
    q.push(1);    // Adds 1 to the queue
    q.push(2);    // Adds 2 to the queue
    q.emplace(4); // Adds 4 to the queue (similar to push but slightly more efficient)

    // Access and modify the back element
    q.back() += 5; // Increment the back element (4) by 5. Now the back becomes 9.

    cout << "The last (back) element after incrementing: " << q.back() << endl;

    // Access the front element
    cout << "The first (front) element: " << q.front() << endl;

    // Remove the front element
    q.pop(); // Removes the front element (1)

    // Access the new front element
    cout << "The new front element after pop: " << q.front() << endl;

    // Check the size of the queue
    cout << "Current size of the queue: " << q.size() << endl;

    // Check if the queue is empty
    cout << "Is the queue empty? " << (q.empty() ? "Yes" : "No") << endl;
}

int main()
{
    cout << "Explaining Queue Operations:\n"
         << endl;
    explainQueue();
    return 0;
}
