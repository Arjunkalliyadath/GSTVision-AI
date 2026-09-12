from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="ModelVersion",
            fields=[
                ("id",             models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("version",        models.CharField(max_length=50)),
                ("trained_at",     models.DateTimeField(auto_now_add=True)),
                ("epochs",         models.IntegerField(default=10)),
                ("training_files", models.IntegerField(default=0)),
                ("final_loss",     models.FloatField(default=0.0)),
                ("accuracy",       models.FloatField(default=0.0)),
                ("is_active",      models.BooleanField(default=True)),
                ("notes",          models.TextField(blank=True)),
            ],
            options={"ordering": ["-trained_at"]},
        ),
        migrations.CreateModel(
            name="AutoTrainingLog",
            fields=[
                ("id",             models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("triggered_at",   models.DateTimeField(auto_now_add=True)),
                ("trigger_reason", models.CharField(max_length=200)),
                ("status",         models.CharField(
                    choices=[("running", "Running"), ("complete", "Complete"), ("failed", "Failed")],
                    default="running", max_length=20)),
                ("model_version",  models.ForeignKey(
                    blank=True, null=True,
                    on_delete=django.db.models.deletion.SET_NULL,
                    to="training.modelversion")),
                ("error_message",  models.TextField(blank=True)),
            ],
            options={"ordering": ["-triggered_at"]},
        ),
    ]
