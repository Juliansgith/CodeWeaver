import unittest
from unittest.mock import patch
from codeweaver.gui.main_window import MainWindow

class TestMainWindow(unittest.TestCase):

    @patch('tkinter.Tk.mainloop')
    def test_main_window_initialization(self, mock_mainloop):
        try:
            app = MainWindow()
            app.destroy()
        except Exception as e:
            self.fail(f"MainWindow initialization failed with {e}")

if __name__ == '__main__':
    unittest.main()