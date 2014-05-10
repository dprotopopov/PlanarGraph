using System.Collections.Generic;
using System.Linq;
using MyLibrary.Types;
using PlanarGraph.Data;

namespace PlanarGraph.Comparer
{
    public class SegmentComparer : IComparer<Segment>, IEqualityComparer<Segment>
    {
        private static readonly VertexComparer VertexComparer = new VertexComparer();

        public int Compare(Segment x, Segment y)
        {
            List<int> l1 = x.Select(v => v.Id).ToList();
            List<int> l2 = y.Select(v => v.Id).ToList();
            l1.Sort();
            l2.Sort();
            int value = l1[0] - l2[0];
            return value != 0 ? value : l1[1] - l2[1];


            //var list1 = new SegmentCollection(x);
            //var list2 = new SegmentCollection(y);
            //var list11 = new StackListQueue<int>(list1.SelectMany(list1.GetInts));
            //var list22 = new StackListQueue<int>(list2.SelectMany(list2.GetInts));
            //try
            //{
            //    int[,] matrix;
            //        StackListQueue<StackListQueue<int>> stackListQueue;
            //        StackListQueue<StackListQueue<int>> stackListQueue2;
            //    {
            //        stackListQueue = new StackListQueue<StackListQueue<int>>(list11) { list11.GetReverse() };
            //        lock (CudafySequencies.Semaphore)
            //        {
            //            var arrayOfArray = stackListQueue.Select(i => i.ToArray()).ToArray();
            //            CudafySequencies.SetSequencies(arrayOfArray, arrayOfArray);
            //            CudafySequencies.Execute("Compare");
            //            matrix = CudafySequencies.GetMatrix();
            //        }
            //        if (matrix[0, 1] > 0) stackListQueue.RemoveAt(0);
            //        else stackListQueue.RemoveAt(1); 
            //    }
            //    {
            //        stackListQueue2 = new StackListQueue<StackListQueue<int>>(list22) { list22.GetReverse() };
            //        lock (CudafySequencies.Semaphore)
            //        {
            //            var arrayOfArray = stackListQueue2.Select(i => i.ToArray()).ToArray();
            //            CudafySequencies.SetSequencies(arrayOfArray, arrayOfArray);
            //            CudafySequencies.Execute("Compare");
            //            matrix = CudafySequencies.GetMatrix();
            //        }
            //        if (matrix[0, 1] > 0) stackListQueue2.RemoveAt(0);
            //        else stackListQueue2.RemoveAt(1);
            //    }
            //    lock (CudafySequencies.Semaphore)
            //    {
            //        CudafySequencies.SetSequencies(stackListQueue.Select(i => i.ToArray()).ToArray(),
            //            stackListQueue2.Select(i => i.ToArray()).ToArray());
            //        CudafySequencies.Execute("Compare");
            //        matrix = CudafySequencies.GetMatrix();
            //    }
            //    return matrix[0, 0];
            //}
            //catch (Exception exception)
            //{
            //    list11.Sort();
            //    list22.Sort();
            //    return
            //        list11.Select((t, i) => t - list22[i])
            //            .FirstOrDefault(compare => compare != 0);
            //}
        }

        public bool Equals(Segment x, Segment y)
        {
            return Compare(x, y) == 0;
        }

        public int GetHashCode(Segment obj)
        {
            return obj.Select(vertex => VertexComparer.GetHashCode(vertex)).Aggregate(Int32.Mul);
        }
    }
}