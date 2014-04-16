using System.Collections.Generic;
using System.Linq;
using PlanarGraph.Data;

namespace PlanarGraph.Comparer
{
    internal class PathComparer : IComparer<Path>
    {
        private static readonly GraphComparer GraphComparer = new GraphComparer();

        public int Compare(Path x, Path y)
        {
            int value = x.Count() - y.Count;
            if (value != 0) return value;
            int count = x.Count;
            if (count == 0) return 0;
            List<int> l1 = x.Select(v => v.Id).ToList();
            List<int> l2 = y.Select(v => v.Id).ToList();
            l1.Sort();
            l2.Sort();
            value = l1.Select((i, j) => i - l2[j]).FirstOrDefault(v => v != 0);
            return value != 0 ? value : GraphComparer.Compare(new Graph(x), new Graph(y));

            //try
            //{
            //    int[,] matrix;
            //    var list1 = new PathCollection(x);
            //    var list11 = new StackListQueue<StackListQueue<int>>(list1.Select(list1.GetInts));
            //    count = list11.Count;
            //    {
            //        var stackListQueue = new StackListQueue<StackListQueue<int>>(list11)
            //        {
            //            list11.Select(p => p.GetReverse())
            //        };
            //        lock (CudafySequencies.Semaphore)
            //        {
            //            int[][] arrayOfArray = stackListQueue.Select(i => i.ToArray()).ToArray();
            //            CudafySequencies.SetSequencies(arrayOfArray, arrayOfArray);
            //            CudafySequencies.Execute("Compare");
            //            matrix = CudafySequencies.GetMatrix();
            //        }
            //        list11 =
            //            new StackListQueue<StackListQueue<int>>(
            //                list11.Select((p, i) => matrix[i, i + count] <= 0 ? p : stackListQueue[i + count]));
            //    }

            //    var list2 = new PathCollection(y);
            //    var list22 = new StackListQueue<StackListQueue<int>>(list2.Select(list2.GetInts));
            //    count = list22.Count;
            //    {
            //        var stackListQueue = new StackListQueue<StackListQueue<int>>(list22)
            //        {
            //            list22.Select(p => p.GetReverse())
            //        };
            //        lock (CudafySequencies.Semaphore)
            //        {
            //            int[][] arrayOfArray = stackListQueue.Select(i => i.ToArray()).ToArray();
            //            CudafySequencies.SetSequencies(arrayOfArray, arrayOfArray);
            //            CudafySequencies.Execute("Compare");
            //            matrix = CudafySequencies.GetMatrix();
            //        }
            //        list22 =
            //            new StackListQueue<StackListQueue<int>>(
            //                list22.Select((p, i) => matrix[i, i + count] <= 0 ? p : stackListQueue[i + count]));
            //    }

            //    lock (CudafySequencies.Semaphore)
            //    {
            //        CudafySequencies.SetSequencies(list11.Select(i => i.ToArray()).ToArray(),
            //            list22.Select(i => i.ToArray()).ToArray());
            //        CudafySequencies.Execute("Compare");
            //        matrix = CudafySequencies.GetMatrix();
            //    }
            //    return matrix[0, 0];
            //}
            //catch (Exception exception)
            //{
            //    return GraphComparer.Compare(new Graph(x), new Graph(y));
            //}
        }
    }
}