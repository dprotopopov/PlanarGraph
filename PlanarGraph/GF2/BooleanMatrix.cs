using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using PlanarGraph.Collections;
using Boolean = PlanarGraph.Types.Boolean;

namespace PlanarGraph.GF2
{
    public class BooleanMatrix : StackListQueue<BooleanVector>
    {
        public BooleanMatrix(IEnumerable<IEnumerable<bool>> list)
        {
            foreach (var item in list)
                Add(new BooleanVector(item));
        }

        public BooleanMatrix(IEnumerable<int> indexes)
        {
            foreach (int item in indexes)
                Add(new BooleanVector(Enumerable.Repeat(false, item))
                {
                    true,
                    Enumerable.Repeat(false, indexes.Count() - item - 1)
                });
            for (int i = 0; i < indexes.Count(); i++) this[i][i] = true;
            Debug.Assert(Count == Length);
        }

        /// <summary>
        ///     Количество столбцов матрицы
        /// </summary>
        public int Length
        {
            get { return this.Max(row => row.Count); }
        }

        /// <summary>
        ///     Определитель булевой матрицы
        ///     Матрица обратима тогда и только тогда, когда определитель матрицы отличен от нуля
        /// </summary>
        public bool Det
        {
            get
            {
                Debug.Assert(Count == Length);
                return !this.Any(vector => vector.IsZero()) && (Count == 1
                    ? this[0][0]
                    : (from index in
                        this[0].Select((b, index) => new KeyValuePair<int, bool>(index, b))
                            .Where(pair => pair.Value)
                            .Select(pair => pair.Key)
                        let vectors = GetRange(1, Count - 1).Select(vector => vector.ToList())
                        select
                            new BooleanMatrix(
                                vectors.Select(
                                    vector =>
                                        new BooleanVector(vector.GetRange(0, index))
                                        {
                                            vector.GetRange(index + 1, Count - index - 1)
                                        }))).Aggregate(false,
                                            (current, submatrix) => Boolean.Xor(current, submatrix.Det)));
            }
        }

        /// <summary>
        ///     Таким образом, каждый базис в этом пространстве получается из данного базиса при помощи цепочки элементарных
        ///     преобразований. А на матричном языке проблема распознавания планарности сводится к нахождению такой матрицы в
        ///     классе эквивалентных матриц (т.е. матриц, которые получаются друг из друга при помощи элементарных преобразований
        ///     над строками), у которой в каждом столбце содержится не более двух единиц [6].
        ///     Указанный критерий позволяет разработать методику определения планарности графа, сводя проблему планарности к
        ///     отысканию минимума некоторого функционала на множестве базисов подпространства квазициклов. Определим следующий
        ///     функционал на матрице С, соответствующий базису подпространства квазициклов (и будем его впредь называть
        ///     функционалом Мак-Лейна)
        ///     Очевидно, что матрица С соответствует базису Мак-Лейна (т.е. базису, удовлетворяющему условию Мак-Лейна) тогда и
        ///     только тогда, когда F(С) = 0.
        /// </summary>
        public int MacLane
        {
            get
            {
                var list = new StackListQueue<int>();
                int length = Length;
                for (int i = 0; i < length; i++)
                {
                    list.Add(this.Count(row => (row.Count > i) && row[i]));
                }
                return list.Sum(s => s*s - 3*s) + 2*length;
            }
        }

        public int E
        {
            get { return this.Sum(item1 => this.Sum(item2 => BooleanVector.Module(item1, item2))); }
        }

        public override string ToString()
        {
            return string.Join(Environment.NewLine, this.Select(item => item.ToString()));
        }
    }
}