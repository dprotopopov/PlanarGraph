using System;
using System.Collections.Generic;
using System.Linq;
using PlanarGraph.Collections;
using PlanarGraph.Comparer;

namespace PlanarGraph.Data
{
    public class Tree : Graph
    {
        private readonly VertexDictionaryComparer _vertexDictionaryComparer = new VertexDictionaryComparer();

        public Tree(Graph graph) : base(graph)
        {
        }

        public Dictionary<KeyValuePair<Vertex, Vertex>, StackListQueue<Path>> GetAllPaths(IEnumerable<Vertex> vertices)
        {
            var stackListQueue =
                new StackListQueue<KeyValuePair<KeyValuePair<Vertex, Vertex>, StackListQueue<Path>>>
                {
                    vertices.Select(
                        vertix =>
                            new KeyValuePair<KeyValuePair<Vertex, Vertex>, StackListQueue<Path>>(
                                new KeyValuePair<Vertex, Vertex>(vertix, vertix),
                                new StackListQueue<Path> {new Path(vertix)}))
                };

            Dictionary<Vertex, VertexEnum> children = ChildrenOrParents;

            for (bool repeat = true;
                repeat;)
            {
                repeat = false;
                for (int i = stackListQueue.Count(); i-- > 0;)
                {
                    KeyValuePair<KeyValuePair<Vertex, Vertex>, StackListQueue<Path>> pair = stackListQueue.Dequeue();
                    stackListQueue.Enqueue(pair);
                    if (_vertexComparer.Compare(pair.Key.Key, pair.Key.Value) > 0) continue;
                    if (!children.ContainsKey(pair.Key.Key)) continue;
                    foreach (Vertex first in children[pair.Key.Key])
                    {
                        if (first.Equals(pair.Key.Key)) continue;
                        if (first.Equals(pair.Key.Value)) continue;
                        IEnumerable<Path> list1 =
                            pair.Value.Where(path => !path.Contains(first)).Select(path => new Path(first) {path});
                        if (!list1.Any()) continue;
                        stackListQueue.Enqueue(new KeyValuePair<KeyValuePair<Vertex, Vertex>, StackListQueue<Path>>(
                            new KeyValuePair<Vertex, Vertex>(first, pair.Key.Value),
                            new StackListQueue<Path> {list1}));
                    }
                }
                for (int i = stackListQueue.Count(); i-- > 0;)
                {
                    KeyValuePair<KeyValuePair<Vertex, Vertex>, StackListQueue<Path>> pair = stackListQueue.Dequeue();
                    stackListQueue.Enqueue(pair);
                    if (_vertexComparer.Compare(pair.Key.Key, pair.Key.Value) > 0) continue;
                    if (!children.ContainsKey(pair.Key.Key)) continue;
                    foreach (Vertex last in children[pair.Key.Value])
                    {
                        if (last.Equals(pair.Key.Key)) continue;
                        if (last.Equals(pair.Key.Value)) continue;
                        IEnumerable<Path> list2 =
                            pair.Value.Where(path => !path.Contains(last)).Select(path => new Path(path) {last});
                        if (!list2.Any()) continue;
                        stackListQueue.Enqueue(new KeyValuePair<KeyValuePair<Vertex, Vertex>, StackListQueue<Path>>(
                            new KeyValuePair<Vertex, Vertex>(pair.Key.Key, last),
                            new StackListQueue<Path> {list2}));
                    }
                }
                for (int i = stackListQueue.Count(); i-- > 0;)
                {
                    KeyValuePair<KeyValuePair<Vertex, Vertex>, StackListQueue<Path>> pair = stackListQueue.Dequeue();
                    stackListQueue.Enqueue(pair);
                    if (pair.Key.Key.Equals(pair.Key.Value)) continue;
                    if (!children.ContainsKey(pair.Key.Key)) continue;
                    stackListQueue.Enqueue(new KeyValuePair<KeyValuePair<Vertex, Vertex>, StackListQueue<Path>>(
                        new KeyValuePair<Vertex, Vertex>(pair.Key.Value, pair.Key.Key),
                        pair.Value));
                }
                if (!stackListQueue.Any()) continue;
                stackListQueue.Sort(_vertexDictionaryComparer);
                KeyValuePair<KeyValuePair<Vertex, Vertex>, StackListQueue<Path>> pair1 = stackListQueue.Dequeue();
                for (int i = stackListQueue.Count(); i-- > 0;)
                {
                    KeyValuePair<KeyValuePair<Vertex, Vertex>, StackListQueue<Path>> pair2 = stackListQueue.Dequeue();
                    if (_vertexDictionaryComparer.Compare(pair1, pair2) == 0)
                    {
                        int count1 = Math.Max(pair1.Value.Count, pair2.Value.Count);
                        pair1 = new KeyValuePair<KeyValuePair<Vertex, Vertex>, StackListQueue<Path>>(
                            pair1.Key, new StackListQueue<Path> {pair1.Value.Union(pair1.Value)});
                        int count2 = Math.Max(pair1.Value.Count, pair2.Value.Count);
                        repeat = repeat || (count1 < count2);
                    }
                    else
                    {
                        stackListQueue.Enqueue(pair1);
                        pair1 = pair2;
                    }
                }
                stackListQueue.Enqueue(pair1);
            }
            return stackListQueue.ToDictionary(pair => pair.Key, pair => pair.Value);
        }

        public class VertexDictionaryComparer :
            IComparer<KeyValuePair<KeyValuePair<Vertex, Vertex>, StackListQueue<Path>>>
        {
            private readonly VertexComparer _vertexComparer = new VertexComparer();

            public int Compare(KeyValuePair<KeyValuePair<Vertex, Vertex>, StackListQueue<Path>> x,
                KeyValuePair<KeyValuePair<Vertex, Vertex>, StackListQueue<Path>> y)
            {
                int value = _vertexComparer.Compare(x.Key.Key, y.Key.Key);
                if (value != 0) return value;
                return _vertexComparer.Compare(x.Key.Value, y.Key.Value);
            }
        }
    }
}