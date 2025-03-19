#include <bits/stdc++.h>
using namespace std;

int getSecondLargest(vector<int> &arr)
{
    int largestE = arr[0];
    int secondL = INT_MIN;
    for (int i = 0; i < arr.size(); i++)
    {
        if (arr[i] > largestE)
        {
            secondL = largestE;
            largestE = arr[i];
        }
        else if (arr[i] < largestE && arr[i] > secondL)
        {
            secondL = arr[i];
        }
    }
    return (secondL != INT_MIN) ? secondL : -1;
}

int main()
{
    int n;
    cin >> n;
    vector<int> arr(n);
    for (int i = 0; i < n; i++)
        cin >> arr[i];
    cout << getSecondLargest(arr) << endl;
    return 0;
}