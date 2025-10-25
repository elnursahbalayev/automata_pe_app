import flet as ft

class SidebarUi:
    def __init__(self, page):
        self.page = page
        self.choice_null = "Keep"
        self.choice_duplicate = "Keep"
        
    def upload_ui(self):
        # Create radio buttons for null values
        null_radio = ft.RadioGroup(
            content=ft.Column([
                ft.Text("Keep or remove null values", size=16),
                ft.Radio(value="Keep", label="Keep"),
                ft.Radio(value="Remove", label="Remove"),
            ]),
            value=self.choice_null,
            on_change=self.on_null_change
        )
        
        # Create radio buttons for duplicate values
        duplicate_radio = ft.RadioGroup(
            content=ft.Column([
                ft.Text("Keep or remove duplicate values", size=16),
                ft.Radio(value="Keep", label="Keep"),
                ft.Radio(value="Remove", label="Remove"),
            ]),
            value=self.choice_duplicate,
            on_change=self.on_duplicate_change
        )
        
        # Return the controls to be added to the sidebar
        return ft.Column([
            null_radio,
            ft.Divider(),
            duplicate_radio
        ])
    
    def on_null_change(self, e):
        self.choice_null = e.control.value
        
    def on_duplicate_change(self, e):
        self.choice_duplicate = e.control.value