from tkinter import Button, Canvas, Frame, Scale, Label, HORIZONTAL, TOP, LEFT
from PIL import Image, ImageTk
from theme import Theme


class ImageDisplayCanvas:
    def __init__(self, parent_frame: Frame, width: int, height: int):
        self.canvas = Canvas(
            parent_frame, 
            bg=Theme.CANVAS_BG, 
            width=width, 
            height=height, 
            relief='sunken', 
            borderwidth=2,
            highlightthickness=0
        )
        self.photo_image_reference = None
        self.original_image = None
        self.scale = 1.0
        self.pan_x = 0
        self.pan_y = 0
        
        # Bind events for zoom and drag
        self.canvas.bind("<MouseWheel>", self.zoom)
        self.canvas.bind("<Button-4>", self.zoom)  # Linux scroll up
        self.canvas.bind("<Button-5>", self.zoom)  # Linux scroll down
        self.canvas.bind("<ButtonPress-1>", self.start_drag)
        self.canvas.bind("<B1-Motion>", self.drag)

    def display_image(self, image):
        """Display image with zoom and pan support"""
        self.original_image = image
        self.scale = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self._render_image()

    def _render_image(self):
        """Internal method to render the image with current zoom level"""
        if self.original_image is None:
            return
            
        # Resize image to fit canvas while maintaining aspect ratio
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width = 800
            canvas_height = 600
        
        img_width, img_height = self.original_image.size
        scale = min(canvas_width / img_width, canvas_height / img_height)
        new_width = int(img_width * scale * self.scale)
        new_height = int(img_height * scale * self.scale)
        
        resized_image = self.original_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        self.photo_image_reference = ImageTk.PhotoImage(resized_image)
        self.canvas.delete("all")
        
        # Center the image with pan offset
        x = canvas_width / 2 + self.pan_x
        y = canvas_height / 2 + self.pan_y
        self.canvas.create_image(x, y, image=self.photo_image_reference)

    def zoom(self, event):
        """Zoom in/out with mouse wheel"""
        if self.original_image is None:
            return
        
        # Determine zoom direction
        if event.num == 5 or event.delta < 0:
            factor = 0.9  # Zoom out
        else:
            factor = 1.1  # Zoom in
        
        self.scale *= factor
        self.scale = max(0.5, min(3.0, self.scale))  # Clamp between 0.5x and 3x
        self._render_image()

    def start_drag(self, event):
        """Start panning the image"""
        self.canvas.scan_mark(event.x, event.y)

    def drag(self, event):
        """Pan the image"""
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    def grid_layout(self, row: int, column: int, columnspan: int = 1, **kwargs):
        self.canvas.grid(row=row, column=column, columnspan=columnspan, **kwargs)

    def pack(self, **kwargs):
        self.canvas.pack(**kwargs)


class ModelButton:
    def __init__(self, parent_frame: Frame, label: str, model_name: str = None, on_click=None):
        self.model_name = model_name
        self.button_color = Theme.get_model_color(model_name) if model_name else Theme.PANEL
        
        self.button = Button(
            parent_frame,
            text=label,
            command=on_click,
            font=("Segoe UI", 11, "bold"),
            relief="flat",
            bg=self.button_color,
            fg="#000000",  # Dark text on bright colored buttons
            activebackground=self.button_color,
            padx=12,
            pady=8,
            cursor="hand2",
            highlightthickness=0,
            borderwidth=0
        )
        
        # Hover effects
        self.button.bind("<Enter>", self.on_hover)
        self.button.bind("<Leave>", self.on_leave)
        self.is_active = False

    def on_hover(self, event):
        """Handle hover effect"""
        if not self.is_active:
            # Darken the button color on hover
            self.button.config(relief="raised", borderwidth=2)

    def on_leave(self, event):
        """Handle leave effect"""
        if not self.is_active:
            self.button.config(relief="flat", borderwidth=0)

    def grid_layout(self, row: int, column: int, padx: int = 5, pady: int = 5, **kwargs):
        self.button.grid(row=row, column=column, padx=padx, pady=pady, **kwargs)

    def set_active(self, is_active: bool):
        """Set button active/inactive state"""
        self.is_active = is_active
        if is_active:
            self.button.config(relief="sunken", borderwidth=3)
        else:
            self.button.config(relief="flat", borderwidth=0)
        
    def pack(self, **kwargs):
        """Pack the button"""
        self.button.pack(**kwargs)


class ParameterSlider:
    def __init__(self, parent_frame: Frame, label: str, min_value: int, max_value: int, initial_value: int):
        self.slider_frame = Frame(parent_frame, relief='sunken', borderwidth=1, bg=Theme.PANEL)
        
        self.label = Label(
            self.slider_frame, 
            text=label, 
            font=("Segoe UI", 10, "bold"), 
            **Theme.get_label_style()
        )
        self.label.pack(side=TOP, pady=5)
        
        # Horizontal frame for slider and value
        controls_frame = Frame(self.slider_frame, bg=Theme.PANEL)
        controls_frame.pack(side=TOP, padx=5, pady=5)
        
        self.slider = Scale(
            controls_frame, 
            from_=min_value, 
            to=max_value, 
            orient=HORIZONTAL, 
            length=180, 
            relief='raised',
            **Theme.get_slider_style()
        )
        self.slider.set(initial_value)
        self.slider.pack(side=LEFT, padx=5)
        
        self.value_label = Label(
            controls_frame, 
            text=str(initial_value), 
            font=("Segoe UI", 11, "bold"), 
            bg=Theme.PANEL, 
            fg=Theme.ACCENT, 
            width=3
        )
        self.value_label.pack(side=LEFT, padx=10)

    def get_value(self) -> int:
        return self.slider.get()

    def set_command(self, callback):
        def update_with_label(value):
            self.value_label.config(text=str(int(float(value))))
            callback(value)
        self.slider.config(command=update_with_label)
    
    def set_enabled(self, enabled: bool):
        """Enable or disable the slider"""
        state = "normal" if enabled else "disabled"
        self.slider.config(state=state)
        self.label.config(fg=Theme.TEXT if enabled else Theme.DISABLED)

    def grid_layout(self, row: int, column: int, **kwargs):
        self.slider_frame.grid(row=row, column=column, sticky='ew', **kwargs)


class ComparisonCanvas:
    """Side-by-side image comparison (Before/After)"""
    
    def __init__(self, parent_frame: Frame, width: int, height: int):
        self.parent_frame = parent_frame
        self.width = width
        self.height = height
        
        # Create container frame
        self.container = Frame(parent_frame, bg=Theme.BG)
        
        # Original image canvas
        self.canvas_original = ImageDisplayCanvas(self.container, width // 2 - 10, height)
        
        # Separator
        separator = Frame(self.container, bg=Theme.ACCENT, width=2)
        separator.pack(side=LEFT, fill='y', padx=5)
        
        # Segmented image canvas
        self.canvas_segmented = ImageDisplayCanvas(self.container, width // 2 - 10, height)
    
    def pack(self, **kwargs):
        """Pack the comparison container"""
        self.container.pack(**kwargs)
        self.canvas_original.pack(side=LEFT, fill='both', expand=True, padx=5)
        self.canvas_segmented.pack(side=LEFT, fill='both', expand=True, padx=5)
    
    def display_images(self, original, segmented):
        """Display both original and segmented images"""
        self.canvas_original.display_image(original)
        self.canvas_segmented.display_image(segmented)
