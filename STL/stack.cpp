#include <bits/stdc++.h>
using namespace std;

void explainStack()
{
    // Create a stack of integers
    stack<int> st;

    // Push elements onto the stack
    st.push(1); // Stack becomes: [1]
    st.push(2); // Stack becomes: [1, 2]
    st.push(3); // Stack becomes: [1, 2, 3]
    st.push(4); // Stack becomes: [1, 2, 3, 4]

    // Emplace is similar to push but constructs the element in place
    st.emplace(5); // Stack becomes: [1, 2, 3, 4, 5]

    // Get the top element of the stack
    cout << st.top() << endl; // Outputs: 5 (top of the stack)

    // Remove the top element
    st.pop();                 // Stack becomes: [1, 2, 3, 4]
    cout << st.top() << endl; // Outputs: 4 (new top of the stack)

    // Get the size of the stack
    cout << st.size() << endl; // Outputs: 4 (number of elements in the stack)

    // Check if the stack is empty
    cout << st.empty() << endl; // Outputs: 0 (false, stack is not empty)

    // Swap contents of two stacks
    stack<int> st1, st2;
    st1.swap(st2); // st1 and st2 are swapped

    // All operations are O(1), no indexing allowed, LIFO Data Structure.
}

int main()
{
    explainStack();
    return 0;
}
