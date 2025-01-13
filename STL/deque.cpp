#include <bits/stdc++.h>
using namespace std;

void printDeque(const deque<int> &dq)
{
    for (int x : dq)
    {
        cout << x << " ";
    }
    cout << endl;
}

void explainDeque()
{
    deque<int> dq;

    // Adding elements
    dq.push_back(2);
    dq.emplace_back(4);
    dq.push_front(5);
    dq.emplace_front(6);
    cout << "After pushing: ";
    printDeque(dq); // Output: 6 5 2 4

    // Removing elements
    dq.pop_back();
    dq.pop_front();
    cout << "After popping: ";
    printDeque(dq); // Output: 5 2

    // Accessing elements
    cout << "Front: " << dq.front() << ", Back: " << dq.back() << endl; // Output: Front: 5, Back: 2

    // rest functions same as vector
    // begin, end, rbegin, rend, clear, insert, size, swap
}

int main()
{
    explainDeque();
    return 0;
}
