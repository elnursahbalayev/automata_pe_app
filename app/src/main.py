import flet as ft


def main(page: ft.Page):
    page.title = "NavigationBar Routing Example"
    page.vertical_alignment = ft.MainAxisAlignment.START

    # --- Define State for Page 2 (Counter) ---
    # We define this control in the main() scope so its state
    # is preserved when we switch pages.
    txt_number = ft.Text("0", size=40, weight=ft.FontWeight.BOLD)

    def minus_click(e):
        txt_number.value = str(int(txt_number.value) - 1)
        page.update()

    def plus_click(e):
        txt_number.value = str(int(txt_number.value) + 1)
        page.update()

    # --- Define Page Content Builders ---
    # These functions create the content for each "page".

    def create_home_view():
        """Content for the Home page."""
        return ft.Column(
            [
                ft.Icon(ft.Icons.HOME, size=100, color=ft.Colors.BLUE_600),
                ft.Text("Welcome Home!", size=30, weight=ft.FontWeight.BOLD),
                ft.Text("This is the main page."),
                ft.Text("Click the icons below to navigate."),
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=20,
            expand=True,
            alignment=ft.MainAxisAlignment.CENTER
        )

    def create_counter_view():
        """Content for the Counter page. It uses the txt_number from main()."""
        return ft.Column(
            [
                ft.Text("Simple Counter", size=30, weight=ft.FontWeight.BOLD),
                ft.Text("This page's state (the number) is preserved."),
                ft.Row(
                    [
                        ft.IconButton(ft.Icons.REMOVE_CIRCLE, on_click=minus_click, icon_size=40,
                                      icon_color=ft.Colors.RED_500),
                        txt_number,  # This is the stateful control from main()
                        ft.IconButton(ft.Icons.ADD_CIRCLE, on_click=plus_click, icon_size=40,
                                      icon_color=ft.Colors.GREEN_500),
                    ],
                    alignment=ft.MainAxisAlignment.CENTER,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER
                ),
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=20,
            expand=True,
            alignment=ft.MainAxisAlignment.CENTER
        )

    def create_settings_view():
        """Content for the Settings page."""
        return ft.Column(
            [
                ft.Text("Settings", size=30, weight=ft.FontWeight.BOLD),
                ft.TextField(label="Username", width=300, icon=ft.Icons.PERSON),
                ft.Checkbox(label="Enable notifications (dummy)"),
                ft.Switch(label="Dark mode (dummy)"),
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=10,
            expand=True,
            alignment=ft.MainAxisAlignment.CENTER
        )

    # --- Navigation and Routing Logic ---

    # A dictionary mapping routes to their content-builder function and nav index
    routes = {
        "/": (create_home_view, 0),
        "/counter": (create_counter_view, 1),
        "/settings": (create_settings_view, 2),
    }

    # A list to easily map a nav index back to a route
    route_list = ["/", "/counter", "/settings"]

    # Create the NavigationBar
    nav_bar = ft.NavigationBar(
        selected_index=0,
        destinations=[
            ft.NavigationBarDestination(icon=ft.Icons.HOME_OUTLINED, selected_icon=ft.Icons.HOME, label="Home"),
            ft.NavigationBarDestination(icon=ft.Icons.CALCULATE_OUTLINED, selected_icon=ft.Icons.CALCULATE,
                                     label="Counter"),
            ft.NavigationBarDestination(icon=ft.Icons.SETTINGS_OUTLINED, selected_icon=ft.Icons.SETTINGS,
                                     label="Settings"),
        ]
    )

    def route_change(route):
        """
        This is the main function that handles navigation.
        It's called every time page.route changes.
        """
        route_path = page.route

        # Default to home if route is not found
        if route_path not in routes:
            route_path = "/"

        # Get the builder function and nav index for the current route
        content_builder, nav_index = routes[route_path]

        # 1. Clear the existing page content
        page.views.clear()

        # 2. Add a new View
        page.views.append(
            ft.View(
                route=route_path,
                controls=[
                    ft.AppBar(title=ft.Text(page.title), bgcolor=ft.Colors.ON_SURFACE_VARIANT),
                    content_builder()  # Call the builder function to get the page content
                ],
                # Add the nav bar to THIS view
                navigation_bar=nav_bar
            )
        )

        # 3. Update the nav bar's selected index
        nav_bar.selected_index = nav_index

        # 4. Update the page
        page.update()

    def nav_change(e):
        """
        This function is called when the user clicks a NavigationBar item.
        It just changes the page route.
        """
        selected_index = e.control.selected_index
        # Use the index to find the corresponding route and go to it
        page.go(route_list[selected_index])

    # --- App Initialization ---

    # Assign the event handlers
    page.on_route_change = route_change
    nav_bar.on_change = nav_change

    # Load the initial route (e.g., "/" or whatever route the user is on)
    page.go(page.route)


# Run the app
if __name__ == "__main__":
    ft.app(target=main)