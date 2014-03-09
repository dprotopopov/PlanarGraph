using System;
using System.Collections.Generic;
using System.Linq;
using PlanarGraph.Collections;

namespace PlanarGraph.GF2
{
    public class BooleanMatrix : StackListQueue<BooleanVector>
    {

        public BooleanMatrix(IEnumerable<IEnumerable<bool>> list)
        {
            foreach(var item in list)
                Add(new BooleanVector(item));
        }

        public int Length
        {
            get { return this.Max(row => row.Count); }
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
        public long MacLane
        {
            get
            {
                var list = new List<long>();
                int length = Length;
                for (int i = 0; i < length; i++)
                {
                    list.Add(this.Count(row => (row.Count > i) && row[i]));
                }
                return list.Sum(s => s*s - 3*s) + 2*length;
            }
        }

        public long E
        {
            get { return this.Sum(item1 => this.Sum(item2 => BooleanVector.Module(item1, item2))); }
        }

        public override string ToString()
        {
            return string.Join(Environment.NewLine, this.Select(item => item.ToString()));
        }
    }
}