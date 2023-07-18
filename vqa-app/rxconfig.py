import reflex as rx

class VqaappConfig(rx.Config):
    pass

config = VqaappConfig(
    app_name="vqa_app",
    db_url="sqlite:///reflex.db",
    env=rx.Env.DEV,
)