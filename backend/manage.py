import asyncio
import json
from pathlib import Path
import typer
from sqlalchemy.ext.asyncio import AsyncSession

import app.db_models # noqa: F401

from app.database import async_session_factory
from app.users.schema import UserCreate
from app.users.service import create_user
from app.users.models import User as UserModel # 타입 힌트를 위해 임포트
from app.styleguides.schemas import StyleGuideCreate
from app.styleguides import service as styleguides_service

cli = typer.Typer()

async def create_admin_runner(name: str, email: str, password: str, db: AsyncSession):
    """비동기 로직을 실행하는 실제 러너 함수"""
    print("--- Admin User Creation ---")
    try:
        user_data = UserCreate(name=name, email=email, password=password)
        
        print(f"Creating admin user '{email}'...")
        # 모든 모델은 app.db_models 임포트 시점에 로드됩니다.
        admin_user: UserModel = await create_user(user_data=user_data, db=db, role="admin")
        
        print("\n✅ Admin user created successfully!")
        print(f"   ID: {admin_user.id}")
        print(f"   Email: {admin_user.email}")
        print(f"   Role: {admin_user.role}")

    except Exception as e:
        # 오류 발생 시 더 자세한 스택 트레이스를 볼 수 있도록 수정하면 디버깅에 좋습니다.
        import traceback
        traceback.print_exc()
        print(f"\n❌ Error creating admin user: {e}")
    finally:
        print("--- Task Finished ---")


@cli.command(name="create-admin")
def createadmin(
    name: str = typer.Option(..., "--name", "-n", help="Admin's full name."),
    email: str = typer.Option(..., "--email", "-e", help="Admin's email address."),
    password: str = typer.Option(..., "--password", "-p", help="Admin's secure password."),
):
    """
    Creates a new user with 'admin' privileges in the database.
    """
    async def main():
        async with async_session_factory() as session:
            await create_admin_runner(name=name, email=email, password=password, db=session)

    asyncio.run(main())
    
@cli.command(name="import-styleguides")
def import_styleguides(
    file: str = typer.Option(..., "--file", "-f", help="Path to JSON file containing styleguides list"),
    mode: str = typer.Option("error", "--mode", help="On duplicate (unique), either 'error' or 'skip'"),
):
    """
    Import up to 100 style guides from a JSON file.

    Supported JSON formats:
    - Legacy format (items):
      { "items": [ {"name":"...","category":"BODY","docs":"..."}, ... ] }
    - JSON format (style_guides):
      { "style_guides": [ {"category":"articles","number":1,"content":["..."],"examples":{...}}, ... ] }
    """
    file_path = Path(file)
    if not file_path.exists():
        print(f"❌ File not found: {file}")
        raise typer.Exit(code=1)

    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"❌ Failed to read/parse JSON: {e}")
        raise typer.Exit(code=1)

    async def runner():
        async with async_session_factory() as session:
            # Prefer new JSON format if present
            if isinstance(payload, dict) and isinstance(payload.get("style_guides"), list):
                result = await styleguides_service.bulk_import_json(session, payload, mode=mode)
            else:
                items_data = payload.get("items", []) if isinstance(payload, dict) else []
                if not isinstance(items_data, list) or not items_data:
                    print("❌ Invalid JSON: expected 'style_guides' or non-empty 'items'")
                    raise typer.Exit(code=1)
                items: list[StyleGuideCreate] = [StyleGuideCreate(**it) for it in items_data]
                result = await styleguides_service.bulk_import(session, items, mode=mode)

            print(f"✅ Import finished: created={result.created}, skipped={result.skipped}, total={result.total}")

    asyncio.run(runner())
    
@cli.command()
def hello() -> None:
    print("Hello World")

if __name__ == "__main__":
    cli()
