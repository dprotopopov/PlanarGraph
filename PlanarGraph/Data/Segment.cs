using System;
using System.Collections.Generic;
using System.Linq;
using PlanarGraph.Collections;

namespace PlanarGraph.Data
{
    /// <summary>
    ///     Класс сегмента графа
    ///     Сегмент содержит две вершины
    /// </summary>
    public class Segment : VertexSortedCollection, IElement
    {
        public Segment(Vertex vertex1, Vertex vertex2) : base(vertex1)
        {
            Add(vertex2);
        }
        public override StackListQueue<int> GetInts(Vertex values)
        {
            return new StackListQueue<int> { values.Id };
        }

        public Segment(IEnumerable<Vertex> list) : base(list)
        {
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
            return graph.Contains(this);
        }

        public bool BelongsTo(Circle circle)
        {
            return circle.Contains(this);
        }

        public bool FromTo(IEnumerable<Vertex> collection)
        {
            return FromTo(collection, collection);
        }

        public bool FromOrTo(IEnumerable<Vertex> collection)
        {
            return collection.Contains(this.First()) || collection.Contains(this.Last());
        }

        public override bool Equals(object obj)
        {
            var segment = obj as Segment;
            return segment != null && base.Equals(segment);
        }

        public override int GetHashCode()
        {
            return base.GetHashCode();
        }

        public bool FromTo(IEnumerable<Vertex> from, IEnumerable<Vertex> to)
        {
            return from.Contains(this.First()) &&
                   to.Contains(this.Last());
        }

        public override string ToString()
        {
            return string.Format("({0})", base.ToString());
        }
    }
}