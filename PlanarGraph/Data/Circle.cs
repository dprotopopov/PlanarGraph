using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using PlanarGraph.Collections;
using PlanarGraph.Comparer;

namespace PlanarGraph.Data
{
    /// <summary>
    ///     Класс цикла в графе
    ///     Цикл содержит неограниченное число вершин
    /// </summary>
    public class Circle : VertexUnsortedCollection, IElement
    {
        private static readonly VertexComparer VertexComparer = new VertexComparer();

        public Circle()
        {
        }

        public Circle(IEnumerable<Vertex> list) : base(list)
        {
        }

        public Circle(Vertex vertex)
            : base(vertex)
        {
        }

        public override string ToString()
        {
            return string.Format("[{0}]", base.ToString());
        }

        /// <summary>
        ///     В задаче построения плоского графа особую роль играют простые циклы. Простые циклы - это квазициклы, у которых
        ///     локальная степень вершин равна двум. Особая роль простых циклов объясняется тем, что границей грани в плоском
        ///     графе, как правило, является простой цикл. Мощность подмножества простых циклов в графе меньше мощности множества
        ///     квазициклов.
        /// </summary>
        public bool IsSimpleCircle()
        {
            return this.Distinct().Count() == Count;
        }

        /// <summary>
        ///     Определение 1. τ-циклом графа называется простой цикл, между двумя любыми несмежными вершинами которого в
        ///     соответствующем графе не существует маршрутов меньшей длины, чем маршруты, принадлежащие данному циклу.
        /// </summary>
        /// <param name="graph"></param>
        /// <param name="cachedGraphAllPaths"></param>
        /// <returns></returns>
        public bool IsTauCircle(Graph graph,
            Dictionary<KeyValuePair<Vertex, Vertex>, PathCollection> cachedGraphAllPaths)
        {
            Dictionary<KeyValuePair<Vertex, Vertex>, PathCollection> dictionary1 = graph.GetMinPaths(this,
                cachedGraphAllPaths);

            Dictionary<KeyValuePair<Vertex, Vertex>, PathCollection> dictionary2 = Graph.GetSubgraphPaths(this,
                cachedGraphAllPaths);

            return !dictionary1.SelectMany(
                pair =>
                    dictionary2.ContainsKey(pair.Key)
                        ? pair.Value.Where(
                            value => value.Count < dictionary2[pair.Key].First().Count)
                        : pair.Value).Any();
        }

        public IEnumerable<Path> Split(Graph graph)
        {
            var list = new List<Path>();
            var indexes = new StackListQueue<int>(this.Intersect(graph.Vertices).Select(v => IndexOf(v)));
            int count = indexes.Count;
            if (count == 0) return list;
            indexes.Sort();
            indexes.Add(indexes[0]);
            Dictionary<Vertex, VertexSortedCollection> children = graph.Children;
            for (int prev = indexes.Dequeue(); indexes.Any(); prev = indexes.Dequeue())
            {
                if (((prev + 1) == indexes[0]) && children[this[prev]].Contains(this[indexes[0]]))
                    continue;
                if (prev < indexes[0]) list.Add(new Path(GetRange(prev, indexes[0] - prev + 1)));
            }
            if (indexes[0] < Count - 1) list.Add(new Path(GetRange(indexes[0], Count - indexes[0])));
            Debug.WriteLineIf(list.Any(), this + " split by " + graph + " is " +
                                          string.Join(",", list.Select(item => item.ToString())));
            return list;
        }

        public bool BelongsTo(Graph graph)
        {
            int count = this.Count(vertex => graph.Vertices.Contains(vertex));
            if (count != Count) return false;
            Vertex first = this.First(vertex => graph.Vertices.Contains(vertex));
            int index = IndexOf(first);
            for (int i = 0; i < count; i++, index++)
            {
                if (!graph.Children[this[(index + count - 1)%count]].Contains(this[(index)%count]))
                    return false;
            }
            return true;
        }

        public bool BelongsTo(Circle circle)
        {
            throw new System.NotImplementedException();
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

        public override bool Equals(object obj)
        {
            var circle = obj as Circle;
            if (circle == null) return false;
            if (Count != circle.Count) return false;
            if (Count == 0) return true;

            var list1 = new SortedStackListQueue<Vertex>(this) {Comparer = VertexComparer};
            var list2 = new SortedStackListQueue<Vertex>(circle) {Comparer = VertexComparer};
            if (!list1.Equals(list2)) return false;

            int index = circle.IndexOf(this[0]);
            var circle1 = new Circle(circle);
            circle1.Rotate(index);
            if (this.SequenceEqual(circle1)) return true;
            circle1.Rotate();
            circle1.Reverse();
            return this.SequenceEqual(circle1);
        }

        public override int GetHashCode()
        {
            return new Graph(this).GetHashCode();
        }
    }
}