using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;

namespace PlanarGraph.Data
{
    /// <summary>
    ///     Класс вершины графа
    ///     В качестве идентификатора вершины используется число типа long
    /// </summary>
    public class Vertex : IElement
    {
        public Vertex(int id)
        {
            Id = id;
        }

        public Vertex(Vertex vertex)
        {
            Id = vertex.Id;
        }

        public int Id { get; private set; }

        public bool BelongsTo(Segment obj)
        {
            return obj.Contains(this);
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

        public bool FromTo(IEnumerable<Vertex> collection)
        {
            throw new NotImplementedException();
        }

        public bool FromOrTo(IEnumerable<Vertex> collection)
        {
            throw new NotImplementedException();
        }

        public bool BelongsTo(Graph graph)
        {
            return graph.Any(BelongsTo);
        }

        public bool BelongsTo(Circle circle)
        {
            return circle.Contains(this);
        }

        public bool BelongsTo(Edge edge)
        {
            return edge.Contains(this);
        }

        public bool BelongsTo(Path obj)
        {
            return obj.Contains(this);
        }

        public override bool Equals(object obj)
        {
            var vertex = obj as Vertex;
            return vertex != null && Id.Equals(vertex.Id);
        }

        public override int GetHashCode()
        {
            return Id.GetHashCode();
        }

        public override string ToString()
        {
            return Id.ToString(CultureInfo.InvariantCulture);
        }
    }
}