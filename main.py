#!/usr/bin/env python3
"""
CodeWeaver - Entry point for the application
"""

from codeweaver.gui import MainWindow


def main():
    app = MainWindow()
    app.mainloop()


if __name__ == "__main__":
    main()