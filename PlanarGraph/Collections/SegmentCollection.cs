using System.Collections.Generic;
using System.Linq;
using MyCudafy.Collections;
using PlanarGraph.Comparer;
using PlanarGraph.Data;

namespace PlanarGraph.Collections
{
    public class SegmentCollection : SortedStackListQueue<Segment>
    {
        private static readonly SegmentComparer SegmentComparer = new SegmentComparer();

        public SegmentCollection(IEnumerable<Segment> segments)
        {
            Comparer = SegmentComparer;
            AddRange(segments);
        }

        public SegmentCollection()
        {
            Comparer = SegmentComparer;
        }

        public SegmentCollection(Segment segment)
        {
            Comparer = SegmentComparer;
            Add(segment);
        }

        public new void Add(Segment segment)
        {
            if (!segment.First().Equals(segment.Last())) AddExcept(segment);
        }

        public new void AddRange(IEnumerable<Segment> segments)
        {
            base.AddRangeExcept(segments.Where(segment => !segment.First().Equals(segment.Last())));
        }

        public override StackListQueue<int> GetInts(Segment values)
        {
            return new StackListQueue<int>(values.Select(value => value.Id));
        }

        public override int GetHashCode()
        {
            return base.GetHashCode();
        }
    }
}