using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using MyCudafy;
using MyCudafy.Collections;
using PlanarGraph.Collections;

namespace PlanarGraph.Data
{
    /// <summary>
    ///     Класс пути в графе
    ///     Путь содержит неограниченное число вершин
    /// </summary>
    public class Path : VertexUnsortedCollection, IElement
    {
        public Path(IEnumerable<Vertex> list)
            : base(list)
        {
        }

        public Path()
        {
        }

        public Path(Vertex vertex)
            : base(vertex)
        {
        }

        public Path(Vertex vertex1, Vertex vertex2)
            : base(vertex1)
        {
            Add(vertex2);
        }

        /// <summary>
        ///     Если все контактные вершины сегмента S имеют номера вершин какой-то грани Γ,
        ///     то мы будем говорить, что грань Γ вмещает этот сегмент и обозначать S⊂Γ
        /// </summary>
        /// <returns></returns>
        public bool FromTo(IEnumerable<Vertex> collection)
        {
            return FromTo(collection, collection);
        }

        public bool FromOrTo(IEnumerable<Vertex> collection)
        {
            throw new NotImplementedException();
        }

        public bool BelongsTo(Circle circle)
        {
            return BelongsTo(new Graph(circle));
        }

        public bool BelongsTo(Edge edge)
        {
            throw new NotImplementedException();
        }

        public bool BelongsTo(Path path)
        {
            throw new NotImplementedException();
        }

        public bool BelongsTo(Segment segment)
        {
            throw new NotImplementedException();
        }

        public bool Contains(Graph graph)
        {
            throw new NotImplementedException();
        }

        public bool Contains(Circle circle)
        {
            throw new NotImplementedException();
        }

        public bool Contains(Edge edge)
        {
            throw new NotImplementedException();
        }

        public bool Contains(Path path)
        {
            throw new NotImplementedException();
        }

        public bool Contains(Segment segment)
        {
            throw new NotImplementedException();
        }

        public bool BelongsTo(Graph graph)
        {
            Dictionary<Vertex, VertexSortedCollection> children = graph.Children;
            return this.All(children.ContainsKey) &&
                   Enumerable.Range(0, Count - 1).All(i => children[this[i]].Contains(this[i + 1]));
        }

        public override StackListQueue<int> GetInts(Vertex values)
        {
            return new StackListQueue<int> {values.Id};
        }

        public new int IndexOf(Vertex item)
        {
            return base.IndexOf(item);
        }

        public static bool IsSimple(IEnumerable<Vertex> path)
        {
            return path.Distinct().Count() == path.Count();
        }

        public override string ToString()
        {
            return string.Format("<{0}>", base.ToString());
        }

        public override bool Equals(object obj)
        {
            var path = obj as Path;
            return path != null && (base.Equals(path) || base.Equals(path.GetReverse()));
        }

        public override int GetHashCode()
        {
            var reverse = new VertexUnsortedCollection {GetReverse()};
            return base.GetHashCode() ^ reverse.GetHashCode();
        }

        /// <summary>
        ///     Определение 3. Две простые цепи, соединяющие х1 и х2, называются непересекающимися (или
        ///     вершинно-непересекающимися), если у них нет общих вершин, отличных от х1 и х2 (и, следовательно, нет общих ребер).
        /// </summary>
        /// <param name="path1"></param>
        /// <param name="path2"></param>
        /// <returns></returns>
        public static bool IsNonIntersected(Path path1, Path path2)
        {
            Debug.Assert(new StackListQueue<Vertex>
            {
                path1.First(),
                path1.Last(),
                path2.First(),
                path2.Last(),
            }.Distinct().Count() == 2);

            IEnumerable<Vertex> intersect =
                path1.GetRange(1, path1.Count - 2)
                    .Intersect(path2.GetRange(1, path2.Count - 2));
            return !intersect.Any();
        }

        public bool FromTo(IEnumerable<Vertex> from, IEnumerable<Vertex> to)
        {
            return from.Contains(this.First()) &&
                   to.Contains(this.Last());
        }

        public static bool IsNoVertix(Path path)
        {
            return path.Count != 1;
        }

        public static bool IsNoCircle(Path path)
        {
            return !path.First().Equals(path.Last());
        }

        public IEnumerable<Path> SplitBy(Segment segment)
        {
            var list = new StackListQueue<Path>();
            StackListQueue<int> indexes;
            try
            {
                int[,] matrix;
                lock (CudafySequencies.Semaphore)
                {
                    CudafySequencies.SetSequencies(
                        segment.Select(GetInts).Select(item => item.ToArray()).ToArray(),
                        GetRange(1, Count - 2).Select(GetInts).Select(item => item.ToArray()).ToArray()
                        );
                    CudafySequencies.Execute("Compare");
                    matrix = CudafySequencies.GetMatrix();
                }
                lock (CudafyMatrix.Semaphore)
                {
                    CudafyMatrix.SetMatrix(matrix);
                    CudafyMatrix.ExecuteRepeatZeroIndexOfZero();
                    indexes = new StackListQueue<int>(CudafyMatrix.GetIndexes()
                        .Where(index => index >= 0)
                        .Select(index => index + 1));
                }
            }
            catch (Exception ex)
            {
                Debug.WriteLine(ex.ToString());
                indexes = new StackListQueue<int>(GetRange(1, Count - 2).Intersect(segment).Select(v => IndexOf(v)));
            }

            indexes.Sort();
            indexes.Prepend(0);
            indexes.Append(Count - 1);
            for (int prev = indexes.Dequeue(); indexes.Any(); prev = indexes.Dequeue())
            {
                if (((prev + 1) == indexes[0])
                    && segment.Contains(this[prev])
                    && segment.Contains(this[indexes[0]]))
                    continue;
                list.Add(new Path(GetRange(prev, indexes[0] - prev + 1)));
            }
            Debug.WriteLineIf(list.Any(), this + " split by " + segment + " is " +
                                          string.Join(",", list.Select(item => item.ToString())));
            return list;
        }

        public static bool IsLong(Path arg)
        {
            return arg.Count > 2;
        }

        public IEnumerable<Path> SplitBy(Vertex vertex)
        {
            int index = IndexOf(vertex);
            return new StackListQueue<Path>(new Path(GetRange(0, index + 1))) {new Path(GetRange(index, Count - index))};
        }
    }
}