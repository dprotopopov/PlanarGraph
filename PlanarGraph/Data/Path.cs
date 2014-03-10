﻿using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using PlanarGraph.Collections;

namespace PlanarGraph.Data
{
    /// <summary>
    ///     Класс пути в графе
    ///     Путь содержит неограниченное число вершин
    /// </summary>
    public class Path : VertexUnsortedCollection, IElement
    {
        public Path(IEnumerable<Vertex> list) : base(list)
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
            Debug.Assert(new List<Vertex>
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

        public bool FromTo(IEnumerable<Vertex> collection)
        {
            return FromTo(collection, collection);
        }

        public bool FromTo(IEnumerable<Vertex> from, IEnumerable<Vertex> to)
        {
            return from.Contains(this.First()) &&
                   to.Contains(this.Last());
        }

        public bool BelongsTo(Circle circle)
        {
            return BelongsTo(new Graph(circle));
        }

        public bool BelongsTo(Edge edge)
        {
            throw new System.NotImplementedException();
        }

        public bool BelongsTo(Path path)
        {
            throw new System.NotImplementedException();
        }

        public bool BelongsTo(Segment segment)
        {
            throw new System.NotImplementedException();
        }

        public bool Contains(Graph graph)
        {
            throw new System.NotImplementedException();
        }

        public bool Contains(Circle circle)
        {
            throw new System.NotImplementedException();
        }

        public bool Contains(Edge edge)
        {
            throw new System.NotImplementedException();
        }

        public bool Contains(Path path)
        {
            throw new System.NotImplementedException();
        }

        public bool Contains(Segment segment)
        {
            throw new System.NotImplementedException();
        }

        public bool BelongsTo(Graph graph)
        {
            Dictionary<Vertex, VertexSortedCollection> children = graph.Children;
            return this.All(children.ContainsKey) &&
                   Enumerable.Range(0, Count - 1).All(i => children[this[i]].Contains(this[i + 1]));
        }

        public IEnumerable<Path> Split(Graph graph)
        {
            Debug.Assert(Count>=2);
            var list = new List<Path>();
            var indexes =
                new StackListQueue<int>(GetRange(1, Count - 2).Intersect(graph.Vertices).Select(v => IndexOf(v)));
            indexes.Sort();
            indexes.Prepend(0);
            indexes.Append(Count - 1);
            Dictionary<Vertex, VertexSortedCollection> children = graph.Children;
            for (int prev = indexes.Dequeue(); indexes.Any(); prev = indexes.Dequeue())
            {
                if (((prev + 1) == indexes[0])
                    && children.ContainsKey(this[prev])
                    && children[this[prev]].Contains(this[indexes[0]]))
                    continue;
                list.Add(new Path(GetRange(prev, indexes[0] - prev + 1)));
            }
            Debug.WriteLineIf(list.Any(), this + " split by " + graph + " is " +
                                          string.Join(",", list.Select(item => item.ToString())));
            return list;
        }

        public static bool IsNoVertix(Path path)
        {
            return path.Count != 1;
        }
        public static bool IsNoCircle(Path path)
        {
            return !path.First().Equals(path.Last());
        }
    }
}