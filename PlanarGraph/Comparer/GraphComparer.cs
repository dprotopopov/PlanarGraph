using System.Collections.Generic;
using System.Linq;
using MyCudafy.Collections;
using MyLibrary.Types;
using PlanarGraph.Data;

namespace PlanarGraph.Comparer
{
    internal class GraphComparer : IComparer<Graph>, IEqualityComparer<Graph>
    {
        private static readonly SegmentComparer SegmentComparer = new SegmentComparer();

        public int Compare(Graph x, Graph y)
        {
            int value = x.Count - y.Count;
            if (value != 0) return value;
            int count = x.Count;
            if (count == 0) return 0;
            List<int> l1 = x.SelectMany(s => s.Select(v => v.Id)).ToList();
            List<int> l2 = y.SelectMany(s => s.Select(v => v.Id)).ToList();
            l1.Sort();
            l2.Sort();
            value = l1.Select((i, j) => i - l2[j]).FirstOrDefault(v => v != 0);
            if (value != 0) return value;
            var list1 = new SortedStackListQueue<Segment>(x) {Comparer = SegmentComparer};
            var list2 = new SortedStackListQueue<Segment>(y) {Comparer = SegmentComparer};
            if (!list1.IsSorted(list1)) list1.Sort(SegmentComparer);
            if (!list2.IsSorted(list2)) list2.Sort(SegmentComparer);
            return
                list1.Select((s, i) => SegmentComparer.Compare(s, list2[i])).FirstOrDefault(compare => compare != 0);

            //try
            //{
            //    int[,] matrix;
            //    int[] array;
            //    int first;

            //    var compare1 = new int[count, count];
            //    var compare2 = new int[count, count];

            //    var stackListQueue1 = new StackListQueue<StackListQueue<int>>(x.Select(x.GetInts));
            //    {
            //        var stackListQueue = new StackListQueue<StackListQueue<int>>(stackListQueue1)
            //        {
            //            stackListQueue1.Select(p => p.GetReverse())
            //        };
            //        lock (CudafySequencies.Semaphore)
            //        {
            //            int[][] arrayOfArray = stackListQueue.Select(i=>i.ToArray()).ToArray();
            //            CudafySequencies.SetSequencies(arrayOfArray, arrayOfArray);
            //            CudafySequencies.Execute("Compare");
            //            matrix = CudafySequencies.GetMatrix();
            //        }
            //        stackListQueue1 =
            //            new StackListQueue<StackListQueue<int>>(
            //                stackListQueue1.Select((p, i) => matrix[i, i + count] <= 0 ? p : stackListQueue[i + count]));
            //        lock (CudafySequencies.Semaphore)
            //        {
            //            int[][] arrayOfArray = stackListQueue1.Select(i => i.ToArray()).ToArray();
            //            CudafySequencies.SetSequencies(arrayOfArray, arrayOfArray);
            //            CudafySequencies.Execute("Compare");
            //            compare1 = CudafySequencies.GetMatrix();
            //        }
            //        lock (CudafyArray.Semaphore)
            //        {
            //            CudafyArray.SetArray(Enumerable.Range(0, count).ToArray());
            //            CudafyArray.SetCompare(compare1);
            //            CudafyArray.OddEvenSort();
            //            array = CudafyArray.GetArray();
            //        }
            //        stackListQueue1 =
            //            new StackListQueue<StackListQueue<int>>(array.Select(i => stackListQueue1.ElementAt(i)));
            //    }

            //    var stackListQueue2 = new StackListQueue<StackListQueue<int>>(x.Select(x.GetInts));
            //    {
            //        var stackListQueue = new StackListQueue<StackListQueue<int>>(stackListQueue2)
            //        {
            //            stackListQueue2.Select(p => p.GetReverse())
            //        };
            //        lock (CudafySequencies.Semaphore)
            //        {
            //            int[][] arrayOfArray = stackListQueue.Select(i => i.ToArray()).ToArray();
            //            CudafySequencies.SetSequencies(arrayOfArray, arrayOfArray);
            //            CudafySequencies.Execute("Compare");
            //            matrix = CudafySequencies.GetMatrix();
            //        }
            //        stackListQueue2 =
            //            new StackListQueue<StackListQueue<int>>(
            //                stackListQueue2.Select((p, i) => matrix[i, i + count] <= 0 ? p : stackListQueue[i + count]));
            //        lock (CudafySequencies.Semaphore)
            //        {
            //            int[][] arrayOfArray = stackListQueue2.Select(i => i.ToArray()).ToArray();
            //            CudafySequencies.SetSequencies(arrayOfArray, arrayOfArray);
            //            CudafySequencies.Execute("Compare");
            //            compare2 = CudafySequencies.GetMatrix();
            //        }

            //        lock (CudafyArray.Semaphore)
            //        {
            //            CudafyArray.SetArray(Enumerable.Range(0, count).ToArray());
            //            CudafyArray.SetCompare(compare2);
            //            CudafyArray.OddEvenSort();
            //            array = CudafyArray.GetArray();
            //        }
            //        stackListQueue2 =
            //            new StackListQueue<StackListQueue<int>>(array.Select(i => stackListQueue2.ElementAt(i)));
            //    }

            //    lock (CudafySequencies.Semaphore)
            //    {
            //        CudafySequencies.SetSequencies(stackListQueue1.Select(i => i.ToArray()).ToArray(),
            //            stackListQueue2.Select(i => i.ToArray()).ToArray());
            //        CudafySequencies.Execute("Compare");
            //        matrix = CudafySequencies.GetMatrix();
            //    }
            //    lock (CudafyMatrix.Semaphore)
            //    {
            //        CudafyMatrix.SetMatrix(matrix);
            //        CudafyMatrix.ExecuteRangeSelectFirstIndexOfNonZero();
            //        first = CudafyMatrix.GetFirst();
            //    }
            //    return (first < 0) ? 0 : matrix[first, first];
            //}
            //catch (Exception exception)
            //{
            //    var list1 = new StackListQueue<Segment>(x);
            //    var list2 = new StackListQueue<Segment>(y);
            //    if (!list1.IsSorted(list1)) list1.Sort(SegmentComparer);
            //    if (!list2.IsSorted(list2)) list2.Sort(SegmentComparer);
            //    return
            //        list1.Select((s, i) => SegmentComparer.Compare(s, list2[i])).FirstOrDefault(compare => compare != 0);
            //}
        }

        public bool Equals(Graph x, Graph y)
        {
            return Compare(x, y) == 0;
        }

        public int GetHashCode(Graph obj)
        {
            return obj.Select((segment, index) => SegmentComparer.GetHashCode(segment)).Aggregate(Int32.Xor);
        }
    }
}