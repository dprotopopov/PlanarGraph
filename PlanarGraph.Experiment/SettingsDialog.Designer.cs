namespace PlanarGraph.Experiment
{
    partial class SettingsDialog
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.groupBox1 = new System.Windows.Forms.GroupBox();
            this.checkBoxEnableCudafy = new System.Windows.Forms.CheckBox();
            this.okButton = new System.Windows.Forms.Button();
            this.groupBox1.SuspendLayout();
            this.SuspendLayout();
            // 
            // groupBox1
            // 
            this.groupBox1.Controls.Add(this.checkBoxEnableCudafy);
            this.groupBox1.Location = new System.Drawing.Point(92, 56);
            this.groupBox1.Name = "groupBox1";
            this.groupBox1.Size = new System.Drawing.Size(358, 157);
            this.groupBox1.TabIndex = 0;
            this.groupBox1.TabStop = false;
            this.groupBox1.Text = "groupBox1";
            // 
            // checkBoxEnableCudafy
            // 
            this.checkBoxEnableCudafy.AutoSize = true;
            this.checkBoxEnableCudafy.Checked = true;
            this.checkBoxEnableCudafy.CheckState = System.Windows.Forms.CheckState.Checked;
            this.checkBoxEnableCudafy.Location = new System.Drawing.Point(101, 74);
            this.checkBoxEnableCudafy.Name = "checkBoxEnableCudafy";
            this.checkBoxEnableCudafy.Size = new System.Drawing.Size(122, 21);
            this.checkBoxEnableCudafy.TabIndex = 0;
            this.checkBoxEnableCudafy.Text = "Enable Cudafy";
            this.checkBoxEnableCudafy.UseVisualStyleBackColor = true;
            // 
            // okButton
            // 
            this.okButton.DialogResult = System.Windows.Forms.DialogResult.OK;
            this.okButton.Location = new System.Drawing.Point(362, 242);
            this.okButton.Name = "okButton";
            this.okButton.Size = new System.Drawing.Size(75, 23);
            this.okButton.TabIndex = 1;
            this.okButton.Text = "OK";
            this.okButton.UseVisualStyleBackColor = true;
            // 
            // SettingsDialog
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(8F, 16F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(536, 301);
            this.Controls.Add(this.okButton);
            this.Controls.Add(this.groupBox1);
            this.Name = "SettingsDialog";
            this.Text = "SettingsDialog";
            this.groupBox1.ResumeLayout(false);
            this.groupBox1.PerformLayout();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.GroupBox groupBox1;
        private System.Windows.Forms.CheckBox checkBoxEnableCudafy;
        private System.Windows.Forms.Button okButton;
    }
}