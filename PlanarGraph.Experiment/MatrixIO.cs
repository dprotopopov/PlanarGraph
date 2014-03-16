using System;
using System.Drawing;
using System.Globalization;
using System.Windows.Forms;

namespace PlanarGraph.Experiment
{
    public class MatrixIO : DataGridView
    {
        #region Private fields

        private const int DimentionSize = 128; //default row and col size
        //array for the data in the grid
        private bool[,] _mData;

        #endregion

        #region Constructors

        public MatrixIO()
        {
            InitializeDataGridView(DimentionSize, DimentionSize);
        }

        public MatrixIO(int nRows, int nCols)
        {
            InitializeDataGridView(nRows, nCols);
        }

        public MatrixIO(Point location, int nRows, int nCols)
        {
            InitializeDataGridView(nRows, nCols);
            Location = location;
        }

        #endregion

        #region Initialisation of the DataGridView

        private void InitializeDataGridView(int rows, int columns)
        {
            AllowUserToAddRows = false;
            AllowUserToDeleteRows = false;
            AllowUserToResizeRows = false;
            EnableHeadersVisualStyles = false;
            SelectionMode = DataGridViewSelectionMode.CellSelect;
            EditMode = DataGridViewEditMode.EditOnKeystroke;
            Name = "dataGridViewMatrix";
            TabIndex = 0;
            RowHeadersWidth = 60;
            //used to attach event-handlers to the events of the editing control(nice name!)
            ColumnHeadersHeightSizeMode = DataGridViewColumnHeadersHeightSizeMode.DisableResizing;
            RowHeadersWidthSizeMode = DataGridViewRowHeadersWidthSizeMode.DisableResizing;
            for (int i = 0; i < columns; i++)
            {
                AddAColumn(i);
            }
            RowHeadersDefaultCellStyle.Padding = new Padding(3); //helps to get rid of the selection triangle?
            for (int i = 0; i < rows; i++)
            {
                AddARow(i);
            }
            for (int i = 0; i < rows && i < columns; i++)
            {
                Rows[i].Cells[i].ReadOnly = true;
            }
            ColumnHeadersDefaultCellStyle.Font = new Font("Verdana", 6F, FontStyle.Bold, GraphicsUnit.Point, 0);
            ColumnHeadersDefaultCellStyle.Alignment = DataGridViewContentAlignment.MiddleCenter;
            ColumnHeadersDefaultCellStyle.BackColor = Color.Gainsboro;
            RowHeadersDefaultCellStyle.Font = new Font("Verdana", 6F, FontStyle.Bold, GraphicsUnit.Point, 0);
            RowHeadersDefaultCellStyle.Alignment = DataGridViewContentAlignment.MiddleCenter;
            RowHeadersDefaultCellStyle.BackColor = Color.Gainsboro;
            ShowEditingIcon = false;
            SelectionMode = DataGridViewSelectionMode.CellSelect;
            CellLeave += MatrixIO_CellChanged;
            CurrentCellDirtyStateChanged += MatrixIO_CellChanged;
        }

        private void MatrixIO_CellChanged(object sender, EventArgs e)
        {
            int r = CurrentCell.RowIndex;
            int c = CurrentCell.ColumnIndex;
            if (r == c && (bool) Rows[r].Cells[c].EditedFormattedValue)
                CurrentCell.Value = false;
            else if (Rows[r].Cells[c].EditedFormattedValue != Rows[c].Cells[r].EditedFormattedValue)
                Rows[c].Cells[r].Value = Rows[r].Cells[c].EditedFormattedValue;
            Refresh();
        }

        private void AddARow(int i)
        {
            var arow = new DataGridViewRow
            {
                HeaderCell =
                {
                    Value = i.ToString(CultureInfo.InvariantCulture)
                }
            };
            Rows.Add(arow);
        }

        private void AddAColumn(int i)
        {
            var acolumn = new DataGridViewCheckBoxColumn
            {
                HeaderText = i.ToString(CultureInfo.InvariantCulture),
                Name = "Column" + i,
                Width = 40,
                SortMode = DataGridViewColumnSortMode.NotSortable
            };
            //make a Style template to be used in the grid
            using (var acell = new DataGridViewCheckBoxCell
            {
                ValueType = typeof (bool),
                TrueValue = true,
                FalseValue = false,
                Style =
                {
                    BackColor = Color.LightCyan,
                    SelectionBackColor = Color.FromArgb(128, 255, 255)
                }
            })
                acolumn.CellTemplate = acell;
            Columns.Add(acolumn);
        }

        public void MakeMatrixTitle(string title)
        {
            TopLeftHeaderCell.Value = title;
            TopLeftHeaderCell.Style.BackColor = Color.AliceBlue;
        }

        #endregion

        #region Properties and property utility functions

        public bool[,] TheData
        {
            get
            {
                _mData = new bool[RowCount, ColumnCount];
                ExtractCheckboxes();
                return _mData;
            }
            set
            {
                int rows = value.GetUpperBound(0) + 1;
                int cols = value.GetUpperBound(1) + 1;
                _mData = new bool[rows, cols];
                _mData = value;
                ResizeOurself(rows, cols);
                FillCheckboxes();
            }
        }

        private void ResizeOurself(int r, int c)
        {
            //adjust rows and cols, do nothing if they equal 
            //
            while (r < RowCount)
            {
                Rows.RemoveAt(RowCount - 1);
            }
            while (r > RowCount)
            {
                AddARow(RowCount);
            }
            while (c < ColumnCount)
            {
                Columns.RemoveAt(ColumnCount - 1);
            }
            while (c > ColumnCount)
            {
                AddAColumn(ColumnCount);
            }
        }

        private void FillCheckboxes() //fill the textboxes
        {
            for (int r = 0; r < RowCount; r++)
            {
                for (int c = 0; c < ColumnCount; c++)
                {
                    Rows[r].Cells[c].Value = _mData[r, c]; //notice r, c
                }
            }
        }

        private void ExtractCheckboxes()
        {
            for (int r = 0; r < RowCount; r++)
            {
                for (int c = 0; c < ColumnCount; c++)
                {
                    _mData[r, c] = (bool) Rows[r].Cells[c].EditedFormattedValue; //notice r, c 
                }
            }
        }

        #endregion
    }
}