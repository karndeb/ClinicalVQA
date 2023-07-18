"""Welcome to Reflex! This file outlines the steps to create a basic app."""
from typing import List, Dict
import reflex as rx
from .utils import *

options: List[str] = ["Donut", "Ernie"]


class State(rx.State):
    """The app state."""

    # The images to show.
    img: list[str]

    # input default str
    text: str = "Type the question you want to get an answer for..."

    # App header
    header: str = "Clinical QandA"

    # default selection
    option: str = "No selection yet."
    # output
    res: str = "Default"

    processing = False
    complete = False

    async def handle_upload(self, files: List[rx.UploadFile]):
        """Handle the upload of file(s). Currently, handles pdf and jpg files
        Args:
            files: The uploaded files.
        """
        for file in files:
            upload_data = await file.read()
            outfile = rx.get_asset_path(file.filename)

            # Save the file.
            with open(outfile, "wb") as file_object:
                file_object.write(upload_data)

            if file.filename.endswith('.pdf'):
                pdf_to_im(rx.get_asset_path(file.filename))

            # Update the img var.
            self.img.append(file.filename)

    def process(self):
        self.processing, self.complete = True, False
        if self.option == "Donut":
            inf = donut_inf(rx.get_asset_path(self.img[0]), self.text)
            yield
            self.res = str(inf)
        else:
            inf = ernie_inf(rx.get_asset_path(self.img[0]), self.text)
            yield
            self.res = str(inf)
        yield
        self.processing, self.complete = False, True


def index():
    """The main view."""
    return rx.center(
        rx.vstack(
            rx.box(rx.image(src='clinicalQA-icon.png', align='center',
                            width="1000px", height="auto"), border_radius="lg", width="20%"),
            rx.heading(State.header, color="blue", font_size='2em'),
            rx.upload(
                rx.vstack(
                    rx.button(
                        "Select File",
                        color="rgb(107,99,246)",
                        bg="white",
                        border=f"1px solid rgb(107,99,246)",
                    ),
                    rx.text(
                        "Drag and drop files here or click to select files"
                    ),
                ),
                border=f"1px dotted rgb(107,99,246)",
                padding="1.5em",
            ),
            rx.button(
                "Upload",
                on_click=lambda: State.handle_upload(rx.upload_files()), bg="orange",
            ),
            rx.foreach(
                State.img, lambda img: rx.image(src=img, width="100px", height="auto",)
            ),
            rx.text(State.text, color_scheme="green"),
            rx.input(on_change=State.set_text, size="sm"),
            rx.select(
                options,
                placeholder="Select the model.",
                on_change=State.set_option,
                color_schemes="twitter",
            ),
            rx.button("Process", on_click=lambda: State.process,
                      is_loading=State.processing, bg="orange",
                      color="black", size="md"),
            rx.cond(
                State.processing,
                rx.spinner(color="orange", size="md"),
            ),
            rx.cond(
                State.complete,
                rx.text(State.res),
            ),
            bg="white",
            shadow="dark-lg",
            border_radius="lg",
            width="30%",
            align_items="center",
            justify_contents="start",
            direction="column",
            spacing="25px",
            padding="2.9em"
        ),
        width="100%",
        height="100vh",
        background="radial-gradient(#387989, #6dd5ed)",
    )


# Add state and page to the app.
app = rx.App(state=State)
app.add_page(index)
app.compile()
