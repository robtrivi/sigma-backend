from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.models import ClassCatalog
from app.schemas.schemas import CatalogClassRead

router = APIRouter(prefix="/catalogs", tags=["catalogs"])


@router.get("/classes", response_model=list[CatalogClassRead])
def list_classes(db: Session = Depends(get_db)) -> list[CatalogClassRead]:
    rows = db.execute(select(ClassCatalog)).scalars().all()
    return [
        CatalogClassRead(
            classId=row.id,
            nombre=row.name,
            colorHex=row.color_hex,
            iconoPrimeNg=row.icono_primeng,
            description=row.description,
        )
        for row in rows
    ]
