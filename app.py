from typing import Tuple, List
import textwrap
import aiocsv
import aiofiles

from textual.app import App, ComposeResult, Binding
from textual.widgets import DataTable, ListView, ListItem, Label, Footer, LoadingIndicator, Static
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from rich.text import Text
import polars as pl

from model import TransactionCategorizer, to_transaction, load_model


class LoadingScreen(ModalScreen):
    def compose(self) -> ComposeResult:
        yield LoadingIndicator()


class MyApp(App):
    CSS_PATH = "style.tcss"

    BINDINGS = [
        Binding("enter", "select_category", "Select Category", priority=True),
        ("k", "move_category_cursor('up')", "Hover Previous Category"),
        ("j", "move_category_cursor('down')", "Hover Next Category"),
        ("ctrl+k", "move_transaction_cursor('up')", "Select Previous Transaction"),
        ("ctrl+j", "move_transaction_cursor('down')", "Select Next Transaction"),
        ("s", "save", "Save"),
        ("q", "quit", "Quit"),
    ]

    def __init__(
        self,
        transaction_categorizer: TransactionCategorizer = None,
        table_data: List[Tuple] = None,
        **kwargs
    ):
        """Initialize the app with optional initial state.

        Args:
            transaction_categorizer: A function that returns a TransactionCategorizer instance
            categories: Set of categories
            table_data: A Dict with `'columns'` (list of column names) and `'rows'` (list of tuples)
        """
        super().__init__(**kwargs)
        self.transaction_categorizer = transaction_categorizer
        self.table_data = table_data

        self.columns = [
            ("Posted Date", 12),
            ("Account", 30),
            ("Description", 75),
            ("Category", 25),
            ("Amount", 10),
        ]

    def __get_column_index(self, column_name: str) -> int:
        for index, (name, _) in enumerate(self.columns):
            if name == column_name:
                return index

        raise ValueError(f"Column '{column_name}' not found")

    def compose(self) -> ComposeResult:
        self.data_table = DataTable(
            id="transaction-table", classes="border", cursor_type="row", zebra_stripes=True)
        self.data_table.can_focus = False
        self.data_table.border_title = "Transactions"

        for column_name, width in self.columns:
            self.data_table.add_column(column_name, width=width)
        self.data_table.add_rows(self.table_data)

        self.list_view = ListView(id="category-list", classes="border")
        self.list_view.border_title = "Category Selector"

        self.row_info = Static(id="transaction-details", classes="border")
        self.row_info.border_title = "Transaction Details"

        with Vertical():
            with Horizontal(id="top-container"):
                yield self.list_view
                yield self.row_info
            yield self.data_table
        yield Footer(compact=True, show_command_palette=False)

    async def on_mount(self) -> None:
        self.query_one("#category-list", ListView).focus()

        if self.transaction_categorizer:
            num_categories = len(self.transaction_categorizer.label_space)
            longest_category = max(
                self.transaction_categorizer.label_space, key=lambda x: len(x))
            self.query_one("#top-container").styles.height = num_categories + 2
            self.query_one(
                "#category-list").styles.width = len(longest_category) + len("(100.00%) ") + 4

        self.log(self.stylesheet)

    async def on_list_view_selected(self, event: ListView.Selected) -> None:
        if event.item:
            cursor_row, _ = self.data_table.cursor_coordinate
            coordinate = (cursor_row, self.__get_column_index("Category"))
            selected_category = self.categories[event.index][1]

            self.data_table.update_cell_at(coordinate, selected_category)
            row = self.data_table.get_row_at(cursor_row)
            transaction = to_transaction(
                date=row[self.__get_column_index("Posted Date")],
                account=row[self.__get_column_index("Account")],
                description=row[self.__get_column_index("Description")],
                amount=row[self.__get_column_index("Amount")],
            )
            self.transaction_categorizer.add(
                transaction=transaction,
                label=selected_category
            )

            self.action_move_transaction_cursor("down")

        self.list_view.focus()

    def to_list_view_label(self, label: str, confidence: float) -> str:
        if confidence < 0.5:
            r = 255
            g = int(255 * (confidence * 2))
            b = 0
        else:
            r = int(255 * (1 - (confidence - 0.5) * 2))
            g = 255
            b = 0
        color = f"#{r:02x}{g:02x}{b:02x}"

        confidence_percentage = f"({(confidence * 100):.2f}%)".ljust(
            len("(100.00%) "))
        return f"[{color}]{confidence_percentage}[/] {label}"

    async def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        cursor_row = event.cursor_row
        row = self.data_table.get_row_at(cursor_row)

        # update transaction details widget
        info_text = ""
        max_width = len(max(self.columns, key=lambda x: len(x[0]))[0]) + 2
        for (col_name, _), value in zip(self.columns, row):
            padding = max_width - len(f"{col_name}:")
            label = f"[u]{col_name}:[/u]" + " " * padding
            value_text = textwrap.fill(str(
                value or ''), width=80 - max_width - 8, subsequent_indent=' ' * (max_width + 8))
            info_text += f"{label}\t{value_text}\n"
        self.row_info.update(info_text)

        # update category list view
        next_description = row[self.__get_column_index("Description")]
        predictions = self.transaction_categorizer.predict(next_description)
        self.categories = [
            (self.to_list_view_label(label, confidence), label) for label, confidence in predictions]

        await self.list_view.clear()
        await self.list_view.extend([ListItem(Label(category_with_confidence))
                                    for category_with_confidence, _ in self.categories])
        self.list_view.index = 0

    async def action_save(self) -> None:
        self.push_screen(LoadingScreen())

        async with aiofiles.open("out.csv", "w", encoding="utf-8", newline="") as f:
            writer = aiocsv.AsyncWriter(f)
            for column, _ in self.columns:
                await writer.writerow([column])
            for row_index in range(len(self.data_table.rows)):
                row = self.data_table.get_row_at(row_index)
                await writer.writerow(row)

        self.pop_screen()

        self.notify("Saved to out.csv")

    def action_move_transaction_cursor(self, direction: str) -> None:
        cursor_row, _ = self.data_table.cursor_coordinate

        if direction == "up":
            self.data_table.move_cursor(row=cursor_row - 1)
            scroll_offset = self.data_table.scroll_offset.y
            if cursor_row - 1 < scroll_offset + 5:
                self.data_table.scroll_up()

        elif direction == "down":
            self.data_table.move_cursor(row=cursor_row + 1)
            scroll_offset = self.data_table.scroll_offset.y
            visible_rows = self.data_table.size.height - 3
            if cursor_row + 1 > scroll_offset + visible_rows - 4:
                self.data_table.scroll_down()

    def action_move_category_cursor(self, direction: str) -> None:
        if direction == "up":
            self.list_view.action_cursor_up()

        elif direction == "down":
            self.list_view.action_cursor_down()

    async def action_select_category(self) -> None:
        self.list_view.action_select_cursor()


def load_input():
    columns = [
        "Posted Date",
        "Account",
        "Description",
        "Category",
        "Amount",
    ]
    df = pl.read_csv("in.csv").select(columns).drop_nulls([
        "Posted Date",
        "Account",
        "Description",
        "Amount",
    ])

    return [tuple(row) for row in df.to_numpy()]


if __name__ == "__main__":
    model = load_model()
    input_data = load_input()

    app = MyApp(transaction_categorizer=model, table_data=input_data)
    app.run()
